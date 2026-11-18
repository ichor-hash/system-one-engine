"""
sys1: Stateful & Self-Routing System One Decision Engine
========================================================
Next-generation low-latency primitive engine (Jev+) with:
- Stateful Recurrent Working Memory (64-float GRUCell)
- Raw Numeric Telemetry Tensor Fusion
- Calibrated Epistemic Uncertainty for System 1 -> System 2 Escalation
"""

from sys1.engine import DecisionResult, SystemOneEngine
from sys1.router import HybridSystemRouter, RoutingDecision
from sys1.calibration import compute_ece, ascii_reliability_diagram

__all__ = [
    "DecisionResult",
    "SystemOneEngine",
    "HybridSystemRouter",
    "RoutingDecision",
    "compute_ece",
    "ascii_reliability_diagram",
]

# Conditionally import model if PyTorch is present
try:
    from sys1.model import SystemOneDecisionEngine, export_to_onnx
    __all__.extend(["SystemOneDecisionEngine", "export_to_onnx"])
except ImportError:
    pass

__version__ = "0.1.0"
