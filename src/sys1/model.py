"""
System One (sys1) Model Architecture
===================================
A stateful, multimodal, self-routing System One Decision Engine with:
1. Recurrent working memory (GRUCell) tracking multi-turn state without prompt bloat.
2. Hybrid numeric + text tensor fusion.
3. Evidential epistemic uncertainty head for automatic System 1 -> System 2 escalation.
4. Baked-in temperature calibration parameters for strict probability alignment.
"""

from typing import Dict, Optional, Tuple, Union
import os

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Fallback dummy class if torch is not yet installed in local environment
    class nn:  # type: ignore
        Module = object

if TORCH_AVAILABLE:
    class NumericProjection(nn.Module):
        """Dedicated MLP projection layer for raw floating-point telemetry/metrics."""
        def __init__(self, in_features: int = 16, out_features: int = 64):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
            self.act = nn.GELU()
            self.norm = nn.LayerNorm(out_features)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: [batch_size, 16]
            return self.norm(self.act(self.linear(x)))

    class SystemOneDecisionEngine(nn.Module):
        """
        Next-Generation Stateful System One Decision Engine.

        Improves on standard Jev:
        - Stateful Recurrent Memory (64-float hidden state across turns)
        - Raw Numeric Telemetry Fusion (16 floats -> 64-dim embedding)
        - Evidential Epistemic Uncertainty for System 1 -> System 2 routing
        - Temperature Calibration parameters baked directly into logits
        """
        def __init__(
            self,
            backbone_name_or_module: Union[str, nn.Module] = "answerdotai/ModernBERT-base",
            hidden_size: Optional[int] = None,
            numeric_dim: int = 16,
            numeric_emb_dim: int = 64,
            memory_dim: int = 64,
            max_choices: int = 64,
            use_dummy_backbone: bool = False,
            vocab_size: int = 50368,
        ):
            super().__init__()
            self.numeric_dim = numeric_dim
            self.numeric_emb_dim = numeric_emb_dim
            self.memory_dim = memory_dim
            self.max_choices = max_choices

            # 1. Text Encoder Backbone
            if use_dummy_backbone:
                # Fast, lightweight dummy backbone for unit tests and local ONNX export (<2s)
                self.hidden_size = hidden_size or 64
                self.backbone = nn.Sequential(
                    nn.Embedding(vocab_size, self.hidden_size),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.GELU(),
                )
                self.is_dummy = True
            elif isinstance(backbone_name_or_module, nn.Module):
                self.backbone = backbone_name_or_module
                self.hidden_size = hidden_size or getattr(self.backbone.config, "hidden_size", 768)
                self.is_dummy = False
            else:
                from transformers import AutoModel, AutoConfig
                self.config = AutoConfig.from_pretrained(backbone_name_or_module)
                self.backbone = AutoModel.from_pretrained(backbone_name_or_module, config=self.config)
                self.hidden_size = hidden_size or self.config.hidden_size
                self.is_dummy = False

            # 2. Numeric Telemetry Projection Layer
            self.numeric_proj = NumericProjection(in_features=numeric_dim, out_features=numeric_emb_dim)

            # 3. Recurrent Working Memory Cell (GRUCell)
            # Ingests: [cls_embedding (hidden_size) + numeric_emb (64)]
            # Updates: prev_state (64) -> next_state (64)
            self.recurrent_cell = nn.GRUCell(
                input_size=self.hidden_size + self.numeric_emb_dim,
                hidden_size=self.memory_dim
            )

            # 4. Multimodal Fusion Vector: [cls_embedding + numeric_emb + next_state]
            self.fused_dim = self.hidden_size + self.numeric_emb_dim + self.memory_dim

            # 5. Output Primitive Heads
            # A. Noul (Binary Guardrails / Prompt Injection) Head
            self.noul_head = nn.Linear(self.fused_dim, 1)

            # B. Choice (API Tool / Intent Routing) Head
            self.choice_head = nn.Linear(self.fused_dim, self.max_choices)

            # C. Score (Continuous / Ordinal 1.0-10.0 Escalation Risk) Head
            self.score_head = nn.Linear(self.fused_dim, 1)

            # D. Evidential Epistemic Uncertainty Head (OOD / Novelty detection)
            self.uncertainty_head = nn.Linear(self.fused_dim, 1)

            # 6. Temperature Calibration Parameters (Bake-in scalars)
            # Default initialized to 1.0; optimized post-training via L-BFGS
            self.T_noul = nn.Parameter(torch.ones(1), requires_grad=False)
            self.T_choice = nn.Parameter(torch.ones(1), requires_grad=False)
            self.T_score = nn.Parameter(torch.ones(1), requires_grad=False)
            self.T_uncertainty = nn.Parameter(torch.ones(1), requires_grad=False)

        def _extract_cls(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            if self.is_dummy:
                # Dummy embedding mean pooling
                emb = self.backbone(input_ids)
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(emb).float()
                sum_emb = torch.sum(emb * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_emb / sum_mask

            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, "last_hidden_state"):
                last_hidden = outputs.last_hidden_state
                # Try pooler_output or CLS token [:, 0, :]
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    return outputs.pooler_output
                # CLS token at position 0
                return last_hidden[:, 0, :]
            elif isinstance(outputs, tuple):
                return outputs[0][:, 0, :]
            return outputs[:, 0, :]

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            numeric_vec: Optional[torch.Tensor] = None,
            prev_state: Optional[torch.Tensor] = None,
            apply_temperature: bool = True,
            return_raw_logits: bool = False,
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass.
            
            Args:
                input_ids: [batch_size, seq_len]
                attention_mask: [batch_size, seq_len]
                numeric_vec: [batch_size, 16] (float telemetry metrics; defaults to zeros)
                prev_state: [batch_size, 64] (recurrent memory vector; defaults to zeros)
                apply_temperature: Whether to divide logits by calibrated T scalars
                return_raw_logits: If True, returns un-normalized logits for training loss functions

            Returns:
                Dictionary with:
                    - noul: [batch_size, 1] probability (or logit)
                    - choice: [batch_size, max_choices] distribution (or logits)
                    - score: [batch_size, 1] continuous score mapped to [1.0, 10.0]
                    - uncertainty: [batch_size, 1] epistemic uncertainty [0.0, 1.0]
                    - next_state: [batch_size, 64] updated recurrent state
            """
            batch_size = input_ids.shape[0]
            device = input_ids.device

            # Handle default inputs
            if numeric_vec is None:
                numeric_vec = torch.zeros(batch_size, self.numeric_dim, device=device, dtype=torch.float32)
            if prev_state is None:
                prev_state = torch.zeros(batch_size, self.memory_dim, device=device, dtype=torch.float32)

            # 1. Text backbone embedding
            cls_emb = self._extract_cls(input_ids, attention_mask)

            # 2. Numeric telemetry embedding
            numeric_emb = self.numeric_proj(numeric_vec)

            # 3. Recurrent state transition
            gru_in = torch.cat([cls_emb, numeric_emb], dim=-1)
            next_state = self.recurrent_cell(gru_in, prev_state)

            # 4. Fused Multimodal Representation
            fused_vec = torch.cat([cls_emb, numeric_emb, next_state], dim=-1)

            # 5. Raw Logits
            noul_logit = self.noul_head(fused_vec)
            choice_logits = self.choice_head(fused_vec)
            score_logit = self.score_head(fused_vec)
            uncertainty_logit = self.uncertainty_head(fused_vec)

            if return_raw_logits:
                return {
                    "noul_logits": noul_logit,
                    "choice_logits": choice_logits,
                    "score_logits": score_logit,
                    "uncertainty_logits": uncertainty_logit,
                    "next_state": next_state,
                }

            # 6. Temperature-scaled calibrated outputs (ONNX graph deployment path)
            t_noul = torch.clamp(self.T_noul, min=1e-3) if apply_temperature else 1.0
            t_choice = torch.clamp(self.T_choice, min=1e-3) if apply_temperature else 1.0
            t_score = torch.clamp(self.T_score, min=1e-3) if apply_temperature else 1.0
            t_unc = torch.clamp(self.T_uncertainty, min=1e-3) if apply_temperature else 1.0

            noul_prob = torch.sigmoid(noul_logit / t_noul)
            choice_probs = F.softmax(choice_logits / t_choice, dim=-1)
            # Map score to [1.0, 10.0] range
            score_val = 1.0 + 9.0 * torch.sigmoid(score_logit / t_score)
            uncertainty_prob = torch.sigmoid(uncertainty_logit / t_unc)

            return {
                "noul": noul_prob,
                "choice": choice_probs,
                "score": score_val,
                "uncertainty": uncertainty_prob,
                "next_state": next_state,
            }

    def export_to_onnx(
        model: nn.Module,
        output_path: str,
        seq_len: int = 32,
        opset_version: int = 17,
    ) -> str:
        """Exports the calibrated SystemOneDecisionEngine to an ONNX graph."""
        model.eval()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        batch_size = 1
        device = next(model.parameters()).device

        dummy_input_ids = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        dummy_numeric_vec = torch.zeros(batch_size, model.numeric_dim, dtype=torch.float32, device=device)
        dummy_prev_state = torch.zeros(batch_size, model.memory_dim, dtype=torch.float32, device=device)

        input_names = ["input_ids", "attention_mask", "numeric_vec", "prev_state"]
        output_names = ["noul", "choice", "score", "uncertainty", "next_state"]

        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "numeric_vec": {0: "batch_size"},
            "prev_state": {0: "batch_size"},
            "noul": {0: "batch_size"},
            "choice": {0: "batch_size"},
            "score": {0: "batch_size"},
            "uncertainty": {0: "batch_size"},
            "next_state": {0: "batch_size"},
        }

        # Wrap wrapper class for ONNX to ensure cleanly indexed tuple outputs
        class ONNXExportWrapper(nn.Module):
            def __init__(self, core_model):
                super().__init__()
                self.core_model = core_model

            def forward(self, input_ids, attention_mask, numeric_vec, prev_state):
                out = self.core_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    numeric_vec=numeric_vec,
                    prev_state=prev_state,
                    apply_temperature=True,
                    return_raw_logits=False,
                )
                return (
                    out["noul"],
                    out["choice"],
                    out["score"],
                    out["uncertainty"],
                    out["next_state"],
                )

        wrapper = ONNXExportWrapper(model)

        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask, dummy_numeric_vec, dummy_prev_state),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        return output_path
