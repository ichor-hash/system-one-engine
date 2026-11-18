"""
Calibration and Reliability Assessment Tools
===========================================
Implements Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
Reliability Diagrams, and PyTorch L-BFGS Temperature Scaling.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def compute_ece(
    confidences: Union[np.ndarray, List[float]],
    predictions: Union[np.ndarray, List[int]],
    ground_truth: Union[np.ndarray, List[int]],
    num_bins: int = 15,
) -> Dict[str, float]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        confidences: 1D array of predicted probabilities/confidence scores [0.0, 1.0]
        predictions: 1D array of predicted class indices or binary outcomes (0 or 1)
        ground_truth: 1D array of true binary/class labels (0 or 1)
        num_bins: Number of equal-width bins (default: 15)

    Returns:
        Dict with "ece", "mce", and "bin_details"
    """
    conf = np.asarray(confidences, dtype=np.float64).flatten()
    pred = np.asarray(predictions, dtype=np.int64).flatten()
    true = np.asarray(ground_truth, dtype=np.int64).flatten()

    if len(conf) != len(pred) or len(pred) != len(true):
        raise ValueError("confidences, predictions, and ground_truth must have equal length.")

    n_samples = len(conf)
    if n_samples == 0:
        return {"ece": 0.0, "mce": 0.0, "bin_details": []}

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_details = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Bins: [lower, upper) for all except the last bin which is [lower, upper]
        if i == num_bins - 1:
            in_bin = (conf >= bin_lower) & (conf <= bin_upper)
        else:
            in_bin = (conf >= bin_lower) & (conf < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(pred[in_bin] == true[in_bin])
            bin_conf = np.mean(conf[in_bin])
            abs_diff = np.abs(bin_acc - bin_conf)
            ece += (bin_count / n_samples) * abs_diff
            mce = max(mce, abs_diff)

            bin_details.append({
                "bin_idx": i,
                "range": (round(float(bin_lower), 3), round(float(bin_upper), 3)),
                "count": int(bin_count),
                "accuracy": round(float(bin_acc), 4),
                "confidence": round(float(bin_conf), 4),
                "error": round(float(abs_diff), 4),
            })
        else:
            bin_details.append({
                "bin_idx": i,
                "range": (round(float(bin_lower), 3), round(float(bin_upper), 3)),
                "count": 0,
                "accuracy": 0.0,
                "confidence": 0.0,
                "error": 0.0,
            })

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_details": bin_details,
    }


def ascii_reliability_diagram(ece_result: Dict[str, any], width: int = 40) -> str:
    """
    Renders an ASCII text-based Reliability Diagram to print in console/logs.
    """
    lines = []
    lines.append("=" * 65)
    lines.append(f"  RELIABILITY DIAGRAM (Expected Calibration Error: {ece_result['ece']:.4f})")
    lines.append("=" * 65)
    lines.append(f"{'Bin Range':<14} | {'Count':<6} | {'Conf':<6} | {'Acc':<6} | Calibration Bar")
    lines.append("-" * 65)

    for b in ece_result.get("bin_details", []):
        rng = f"[{b['range'][0]:.2f}, {b['range'][1]:.2f}]"
        cnt = b["count"]
        conf = b["confidence"]
        acc = b["accuracy"]

        if cnt == 0:
            bar = "(empty)"
        else:
            # Bar displays Accuracy (A) vs Confidence (C)
            # perfect calibration is when A and C overlap
            c_pos = int(conf * width)
            a_pos = int(acc * width)
            row = [" "] * (width + 1)
            row[min(c_pos, width)] = "C"
            row[min(a_pos, width)] = "A" if a_pos != c_pos else "*"
            bar = "".join(row)

        lines.append(f"{rng:<14} | {cnt:<6} | {conf:<6.2f} | {acc:<6.2f} | {bar}")

    lines.append("-" * 65)
    lines.append("Legend: '*' = Perfect match | 'C' = Avg Confidence | 'A' = Real Accuracy")
    lines.append("=" * 65)
    return "\n".join(lines)


if TORCH_AVAILABLE:
    class TemperatureScaler(nn.Module):
        """
        Post-training Temperature Scaling optimizer using L-BFGS.
        Minimizes negative log likelihood on validation logits.
        """
        def __init__(self):
            super().__init__()
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        def forward(self, logits: torch.Tensor) -> torch.Tensor:
            return logits / torch.clamp(self.temperature, min=1e-3)

        def fit(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            is_multiclass: bool = False,
            max_iter: int = 50,
            lr: float = 0.01,
        ) -> float:
            """
            Optimizes temperature T on validation logits.
            
            Args:
                logits: [N, C] or [N, 1] tensor of uncalibrated model logits
                labels: [N] true integer labels or [N, 1] floats
                is_multiclass: True for softmax, False for binary cross entropy
            """
            device = logits.device
            self.to(device)

            if is_multiclass:
                nll_criterion = nn.CrossEntropyLoss()
            else:
                nll_criterion = nn.BCEWithLogitsLoss()

            optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

            def eval_loss():
                optimizer.zero_grad()
                scaled_logits = self.forward(logits)
                if is_multiclass:
                    loss = nll_criterion(scaled_logits, labels.long())
                else:
                    loss = nll_criterion(scaled_logits.squeeze(-1), labels.float())
                loss.backward()
                return loss

            optimizer.step(eval_loss)
            return float(self.temperature.item())
