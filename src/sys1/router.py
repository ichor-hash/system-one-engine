"""
Hybrid System 1 -> System 2 Router
==================================
Routes high-frequency agentic queries through the sub-15ms System 1 model.
Automatically escalates out-of-distribution or highly ambiguous edge cases
to an expensive System 2 reasoning LLM based on evidential epistemic uncertainty.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
import time
import numpy as np

from sys1.engine import DecisionResult, SystemOneEngine


@dataclass
class RoutingDecision:
    """Detailed summary of the routing trajectory and execution path."""
    engine_used: str                           # 'system1' or 'system2'
    result: DecisionResult                     # The final decision primitive
    escalated: bool                            # Whether System 2 fallback was triggered
    escalation_reason: Optional[str]           # Justification for escalation (or None)
    system1_latency_ms: float                  # Time spent in System 1 (~5-12ms)
    total_latency_ms: float                    # Total elapsed time including LLM if escalated
    estimated_cost_multiplier: float           # Relative compute cost (1x for sys1, ~50x-100x for LLM)


def default_mock_system2_llm(
    task: str,
    context: str,
    query_or_options: Any,
    metrics: Optional[List[float]] = None,
    state: Optional[np.ndarray] = None,
) -> DecisionResult:
    """
    Simulated System 2 Frontier Reasoning Model (e.g. Claude 3.5 Sonnet / GPT-4o).
    Simulates deliberate reasoning delay (~400-800ms) with near-certain confidence.
    """
    time.sleep(0.05) # Simulated network/generation delay for testing
    
    if task == "noul":
        # System 2 resolves edge cases accurately
        val = True
        return DecisionResult(
            task="noul",
            value=val,
            confidence=0.98,
            probabilities={"positive": 0.98, "negative": 0.02},
            uncertainty=0.05,
            state=state if state is not None else np.zeros(64, dtype=np.float32),
            latency_ms=50.0,
        )
    elif task == "choice":
        options = query_or_options if isinstance(query_or_options, list) else ["option_1", "option_2"]
        chosen = options[0]
        prob_dict = {opt: (0.90 if opt == chosen else 0.10 / (len(options) - 1)) for opt in options}
        return DecisionResult(
            task="choice",
            value=chosen,
            confidence=0.90,
            probabilities=prob_dict,
            uncertainty=0.08,
            state=state if state is not None else np.zeros(64, dtype=np.float32),
            latency_ms=50.0,
        )
    else: # score
        return DecisionResult(
            task="score",
            value=8.5,
            confidence=0.95,
            probabilities=8.5,
            uncertainty=0.05,
            state=state if state is not None else np.zeros(64, dtype=np.float32),
            latency_ms=50.0,
        )


class HybridSystemRouter:
    """
    Orchestrates the System 1 (Intuitive/Fast) and System 2 (Reasoning/Slow) pipeline.
    
    Attributes:
        engine: The initialized SystemOneEngine instance.
        system2_fallback: Optional callable simulating or invoking the System 2 LLM.
        uncertainty_threshold: Epistemic score above which query is escalated (default: 0.65).
        ambiguity_margin: Margin around 0.50 binary probability that triggers escalation.
    """
    def __init__(
        self,
        engine: SystemOneEngine,
        system2_fallback: Optional[Callable[..., DecisionResult]] = None,
        uncertainty_threshold: float = 0.65,
        ambiguity_margin: float = 0.08,
    ):
        self.engine = engine
        self.system2_fallback = system2_fallback or default_mock_system2_llm
        self.uncertainty_threshold = uncertainty_threshold
        self.ambiguity_margin = ambiguity_margin

    def route_noul(
        self,
        context: str,
        question: str,
        metrics: Optional[List[float]] = None,
        state: Optional[np.ndarray] = None,
    ) -> RoutingDecision:
        """Evaluates a binary guardrail query through the hybrid pipeline."""
        t_start = time.perf_counter()
        sys1_res = self.engine.noul(context, question, metrics=metrics, state=state)
        sys1_elapsed = (time.perf_counter() - t_start) * 1000.0

        # Escalation Checks:
        escalate = False
        reason = None

        if sys1_res.uncertainty >= self.uncertainty_threshold:
            escalate = True
            reason = f"High epistemic uncertainty ({sys1_res.uncertainty:.3f} >= {self.uncertainty_threshold:.2f}) [Novelty/OOD]"
        elif abs(sys1_res.confidence - 0.50) < self.ambiguity_margin:
            escalate = True
            reason = f"Ambiguous binary boundary (confidence {sys1_res.confidence:.3f} near 0.50)"

        if escalate and self.system2_fallback:
            sys2_res = self.system2_fallback("noul", context, question, metrics=metrics, state=sys1_res.state)
            total_elapsed = (time.perf_counter() - t_start) * 1000.0
            return RoutingDecision(
                engine_used="system2",
                result=sys2_res,
                escalated=True,
                escalation_reason=reason,
                system1_latency_ms=round(sys1_elapsed, 2),
                total_latency_ms=round(total_elapsed, 2),
                estimated_cost_multiplier=50.0,
            )

        return RoutingDecision(
            engine_used="system1",
            result=sys1_res,
            escalated=False,
            escalation_reason=None,
            system1_latency_ms=round(sys1_elapsed, 2),
            total_latency_ms=round(sys1_elapsed, 2),
            estimated_cost_multiplier=1.0,
        )

    def route_choice(
        self,
        context: str,
        options: List[str],
        metrics: Optional[List[float]] = None,
        state: Optional[np.ndarray] = None,
    ) -> RoutingDecision:
        """Routes a multi-choice tool/intent query through the hybrid pipeline."""
        t_start = time.perf_counter()
        sys1_res = self.engine.choice(context, options, metrics=metrics, state=state)
        sys1_elapsed = (time.perf_counter() - t_start) * 1000.0

        escalate = False
        reason = None

        if sys1_res.uncertainty >= self.uncertainty_threshold:
            escalate = True
            reason = f"High epistemic uncertainty ({sys1_res.uncertainty:.3f} >= {self.uncertainty_threshold:.2f}) [Novelty/OOD]"
        elif sys1_res.confidence < 0.35 and len(options) > 2:
            escalate = True
            reason = f"Low top-1 choice confidence ({sys1_res.confidence:.3f} < 0.35)"

        if escalate and self.system2_fallback:
            sys2_res = self.system2_fallback("choice", context, options, metrics=metrics, state=sys1_res.state)
            total_elapsed = (time.perf_counter() - t_start) * 1000.0
            return RoutingDecision(
                engine_used="system2",
                result=sys2_res,
                escalated=True,
                escalation_reason=reason,
                system1_latency_ms=round(sys1_elapsed, 2),
                total_latency_ms=round(total_elapsed, 2),
                estimated_cost_multiplier=50.0,
            )

        return RoutingDecision(
            engine_used="system1",
            result=sys1_res,
            escalated=False,
            escalation_reason=None,
            system1_latency_ms=round(sys1_elapsed, 2),
            total_latency_ms=round(sys1_elapsed, 2),
            estimated_cost_multiplier=1.0,
        )

    def route_score(
        self,
        context: str,
        rubric: str,
        metrics: Optional[List[float]] = None,
        state: Optional[np.ndarray] = None,
    ) -> RoutingDecision:
        """Routes a scoring/risk telemetry query through the hybrid pipeline."""
        t_start = time.perf_counter()
        sys1_res = self.engine.score(context, rubric, metrics=metrics, state=state)
        sys1_elapsed = (time.perf_counter() - t_start) * 1000.0

        escalate = False
        reason = None

        if sys1_res.uncertainty >= self.uncertainty_threshold:
            escalate = True
            reason = f"High epistemic uncertainty ({sys1_res.uncertainty:.3f} >= {self.uncertainty_threshold:.2f}) [Novelty/OOD]"

        if escalate and self.system2_fallback:
            sys2_res = self.system2_fallback("score", context, rubric, metrics=metrics, state=sys1_res.state)
            total_elapsed = (time.perf_counter() - t_start) * 1000.0
            return RoutingDecision(
                engine_used="system2",
                result=sys2_res,
                escalated=True,
                escalation_reason=reason,
                system1_latency_ms=round(sys1_elapsed, 2),
                total_latency_ms=round(total_elapsed, 2),
                estimated_cost_multiplier=50.0,
            )

        return RoutingDecision(
            engine_used="system1",
            result=sys1_res,
            escalated=False,
            escalation_reason=None,
            system1_latency_ms=round(sys1_elapsed, 2),
            total_latency_ms=round(sys1_elapsed, 2),
            estimated_cost_multiplier=1.0,
        )
