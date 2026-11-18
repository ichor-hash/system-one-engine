"""
SystemOneEngine: High-Performance ONNXRuntime CPU Inference SDK
===============================================================
Delivers sub-15ms calibrated inference for stateful agentic workflows.
Requires zero PyTorch dependencies at runtime.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import time
import os
import numpy as np

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


@dataclass
class DecisionResult:
    """Standardized output container for System One decisions."""
    task: str                                  # 'noul', 'choice', 'score'
    value: Union[bool, str, float]             # Decision primitive value
    confidence: float                          # Calibrated confidence [0.0, 1.0]
    probabilities: Union[Dict[str, float], float]  # Option distributions or binary probability
    uncertainty: float                         # Calibrated Epistemic Novelty score [0.0, 1.0]
    state: np.ndarray                          # Recurrent 64-dim working memory for next turn
    latency_ms: float                          # Wall-clock inference time in milliseconds
    raw_outputs: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)

    def should_escalate(self, uncertainty_threshold: float = 0.65, ambiguity_margin: float = 0.10) -> bool:
        """Determines if this query should fall back to a System 2 reasoning LLM."""
        # 1. High Epistemic Uncertainty (novelty / out-of-distribution)
        if self.uncertainty >= uncertainty_threshold:
            return True

        # 2. Ambiguity (confidence hovering near 50% boundary for binary tasks)
        if self.task == "noul" and abs(self.confidence - 0.50) < ambiguity_margin:
            return True

        return False


class SimpleTokenizerFallback:
    """Lightweight hash-based tokenizer fallback when HuggingFace tokenizers is absent."""
    def __init__(self, max_length: int = 64, vocab_size: int = 50368):
        self.max_length = max_length
        self.vocab_size = vocab_size

    def encode(self, text: str) -> Dict[str, np.ndarray]:
        words = text.strip().split()
        # Simple deterministic hashing into token ids
        token_ids = [101] # [CLS]
        for w in words:
            token_ids.append(abs(hash(w)) % (self.vocab_size - 1000) + 1000)
        token_ids.append(102) # [SEP]

        token_ids = token_ids[:self.max_length]
        pad_len = self.max_length - len(token_ids)
        mask = [1] * len(token_ids) + [0] * pad_len
        token_ids = token_ids + [0] * pad_len

        return {
            "input_ids": np.array([token_ids], dtype=np.int64),
            "attention_mask": np.array([mask], dtype=np.int64),
        }


class MockInferenceSession:
    """Deterministic mock session for zero-dependency local testing & simulation."""
    def __init__(self, memory_dim: int = 64, max_choices: int = 64):
        self.memory_dim = memory_dim
        self.max_choices = max_choices

    def run(self, output_names: Optional[List[str]], input_feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
        input_ids = input_feed.get("input_ids", np.zeros((1, 32), dtype=np.int64))
        batch_size = input_ids.shape[0]
        numeric_vec = input_feed.get("numeric_vec", np.zeros((batch_size, 16), dtype=np.float32))
        prev_state = input_feed.get("prev_state", np.zeros((batch_size, self.memory_dim), dtype=np.float32))

        # Deterministic pseudo-random generation based on input content hash
        content_hash = int(np.sum(input_ids) + np.sum(numeric_vec * 100)) % 10000
        rng = np.random.default_rng(seed=content_hash)

        # Update recurrent state with exponential decay + incoming signal
        next_state = 0.8 * prev_state + 0.2 * rng.standard_normal((batch_size, self.memory_dim), dtype=np.float32)

        # Output heads
        noul = np.clip(rng.uniform(0.1, 0.95, size=(batch_size, 1)).astype(np.float32), 0.0, 1.0)
        choice_raw = rng.uniform(0.01, 1.0, size=(batch_size, self.max_choices)).astype(np.float32)
        choice = choice_raw / np.sum(choice_raw, axis=-1, keepdims=True)
        score = np.clip(rng.uniform(1.0, 10.0, size=(batch_size, 1)).astype(np.float32), 1.0, 10.0)
        uncertainty = np.clip(rng.uniform(0.05, 0.85, size=(batch_size, 1)).astype(np.float32), 0.0, 1.0)

        return [noul, choice, score, uncertainty, next_state]


class SystemOneEngine:
    """
    Sub-15ms System One Inference Engine powered by ONNXRuntime.

    Attributes:
        model_path: Path to the exported INT8 or FP32 .onnx model.
        num_threads: Number of CPU threads dedicated to ONNXRuntime intra-op execution.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        num_threads: int = 4,
        memory_dim: int = 64,
        numeric_dim: int = 16,
        max_choices: int = 64,
        mock_mode: bool = False,
    ):
        self.memory_dim = memory_dim
        self.numeric_dim = numeric_dim
        self.max_choices = max_choices
        self.model_path = model_path
        self.num_threads = num_threads

        # 1. Initialize Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            try:
                from transformers import AutoTokenizer
                # Try loading ModernBERT or fallback
                self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
            except Exception:
                self.tokenizer = SimpleTokenizerFallback()

        # 2. Initialize ONNXRuntime Session
        if mock_mode or not ORT_AVAILABLE or model_path is None or not os.path.exists(model_path):
            self.session = MockInferenceSession(memory_dim=memory_dim, max_choices=max_choices)
            self.is_mock = True
        else:
            session_opts = ort.SessionOptions()
            session_opts.intra_op_num_threads = num_threads
            session_opts.inter_op_num_threads = 1
            session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                model_path,
                sess_options=session_opts,
                providers=["CPUExecutionProvider"]
            )
            self.is_mock = False

    def _prepare_inputs(
        self,
        text: str,
        metrics: Optional[List[float]] = None,
        state: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        # 1. Tokenize text
        if hasattr(self.tokenizer, "encode") and not hasattr(self.tokenizer, "batch_encode_plus"):
            # Simple fallback tokenizer
            enc = self.tokenizer.encode(text)
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
        else:
            # HuggingFace tokenizer
            enc = self.tokenizer(
                text,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="np"
            )
            input_ids = enc["input_ids"].astype(np.int64)
            attention_mask = enc["attention_mask"].astype(np.int64)

        # 2. Format metrics vector
        if metrics is None:
            numeric_vec = np.zeros((1, self.numeric_dim), dtype=np.float32)
        else:
            m_arr = np.array(metrics, dtype=np.float32).flatten()
            if len(m_arr) < self.numeric_dim:
                m_arr = np.pad(m_arr, (0, self.numeric_dim - len(m_arr)))
            else:
                m_arr = m_arr[:self.numeric_dim]
            numeric_vec = m_arr.reshape(1, self.numeric_dim)

        # 3. Format state vector
        if state is None:
            prev_state = np.zeros((1, self.memory_dim), dtype=np.float32)
        else:
            s_arr = np.array(state, dtype=np.float32).flatten()
            if len(s_arr) < self.memory_dim:
                s_arr = np.pad(s_arr, (0, self.memory_dim - len(s_arr)))
            else:
                s_arr = s_arr[:self.memory_dim]
            prev_state = s_arr.reshape(1, self.memory_dim)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "numeric_vec": numeric_vec,
            "prev_state": prev_state,
        }

    def _run_forward(self, feed: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
        t0 = time.perf_counter()
        if self.is_mock:
            raw_out = self.session.run(None, feed)
        else:
            raw_out = self.session.run(
                ["noul", "choice", "score", "uncertainty", "next_state"],
                feed
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        outputs = {
            "noul": raw_out[0],
            "choice": raw_out[1],
            "score": raw_out[2],
            "uncertainty": raw_out[3],
            "next_state": raw_out[4],
        }
        return outputs, elapsed_ms

    def noul(
        self,
        context: str,
        question: str,
        metrics: Optional[List[float]] = None,
        state: Optional[np.ndarray] = None,
        threshold: float = 0.50,
    ) -> DecisionResult:
        """
        Binary Guardrail & Decision Primitive (Yes / No).
        Used for prompt injection checks, safety guardrails, policy adherence.
        """
        full_text = f"Context: {context}\nQuestion: {question}"
        feed = self._prepare_inputs(full_text, metrics=metrics, state=state)
        outputs, latency_ms = self._run_forward(feed)

        prob = float(outputs["noul"][0, 0])
        val = bool(prob >= threshold)
        uncertainty = float(outputs["uncertainty"][0, 0])
        next_state = outputs["next_state"][0]

        return DecisionResult(
            task="noul",
            value=val,
            confidence=prob if val else (1.0 - prob),
            probabilities={"positive": prob, "negative": 1.0 - prob},
            uncertainty=uncertainty,
            state=next_state,
            latency_ms=round(latency_ms, 2),
            raw_outputs=outputs,
        )

    def choice(
        self,
        context: str,
        options: List[str],
        metrics: Optional[List[float]] = None,
        state: Optional[np.ndarray] = None,
    ) -> DecisionResult:
        """
        Multi-Class Tool & Intent Routing Primitive.
        Selects the best option from a list of candidates.
        """
        formatted_options = ", ".join([f"[{i}] {opt}" for i, opt in enumerate(options)])
        full_text = f"Context: {context}\nOptions: {formatted_options}"
        feed = self._prepare_inputs(full_text, metrics=metrics, state=state)
        outputs, latency_ms = self._run_forward(feed)

        num_options = min(len(options), self.max_choices)
        choice_probs = outputs["choice"][0, :num_options]
        # Re-normalize over active choices
        sum_p = np.sum(choice_probs)
        if sum_p > 1e-9:
            choice_probs = choice_probs / sum_p

        best_idx = int(np.argmax(choice_probs))
        best_opt = options[best_idx]
        best_conf = float(choice_probs[best_idx])
        uncertainty = float(outputs["uncertainty"][0, 0])
        next_state = outputs["next_state"][0]

        prob_dict = {options[i]: float(choice_probs[i]) for i in range(num_options)}

        return DecisionResult(
            task="choice",
            value=best_opt,
            confidence=best_conf,
            probabilities=prob_dict,
            uncertainty=uncertainty,
            state=next_state,
            latency_ms=round(latency_ms, 2),
            raw_outputs=outputs,
        )

    def score(
        self,
        context: str,
        rubric: str,
        metrics: Optional[List[float]] = None,
        state: Optional[np.ndarray] = None,
    ) -> DecisionResult:
        """
        Continuous / Ordinal Score Primitive (1.0 to 10.0).
        Used for customer frustration, sentiment drift, fraud severity.
        """
        full_text = f"Context: {context}\nRubric: {rubric}"
        feed = self._prepare_inputs(full_text, metrics=metrics, state=state)
        outputs, latency_ms = self._run_forward(feed)

        score_val = float(np.clip(outputs["score"][0, 0], 1.0, 10.0))
        uncertainty = float(outputs["uncertainty"][0, 0])
        next_state = outputs["next_state"][0]

        return DecisionResult(
            task="score",
            value=round(score_val, 2),
            confidence=1.0 - uncertainty,
            probabilities=score_val,
            uncertainty=uncertainty,
            state=next_state,
            latency_ms=round(latency_ms, 2),
            raw_outputs=outputs,
        )
