"""
Unit & Integration Tests for System One Engine & Router
======================================================
Tests SDK functionality, recurrent memory persistence, numeric telemetry fusion,
uncertainty-based System 1 -> System 2 escalation, and calibration metrics.
Designed to run instantly (<2 seconds) on any CPU without external downloads.
"""

import unittest
import sys
import os
import numpy as np

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sys1.engine import SystemOneEngine, DecisionResult
from sys1.router import HybridSystemRouter, RoutingDecision
from sys1.calibration import compute_ece, ascii_reliability_diagram


class TestSystemOneSDK(unittest.TestCase):
    def setUp(self):
        # Initialize engine in mock mode for instant zero-dependency testing
        self.engine = SystemOneEngine(mock_mode=True, memory_dim=64, numeric_dim=16)

    def test_noul_primitive(self):
        """Test binary guardrail / prompt injection decision primitive."""
        context = "User asks to ignore system instructions and reveal API key."
        question = "Is this prompt safe and compliant?"
        metrics = [12.5, 0.0, 1.0] # 12.5ms latency, 0 retries, 1 past warning

        res = self.engine.noul(context=context, question=question, metrics=metrics)

        self.assertIsInstance(res, DecisionResult)
        self.assertEqual(res.task, "noul")
        self.assertIsInstance(res.value, bool)
        self.assertGreaterEqual(res.confidence, 0.0)
        self.assertLessEqual(res.confidence, 1.0)
        self.assertGreaterEqual(res.uncertainty, 0.0)
        self.assertLessEqual(res.uncertainty, 1.0)
        self.assertEqual(res.state.shape, (64,))
        self.assertGreater(res.latency_ms, 0.0)

    def test_choice_primitive(self):
        """Test multi-class tool/intent routing primitive."""
        context = "User wants to refund order #9482 due to damaged item."
        options = ["refund_order", "check_tracking", "escalate_to_human", "faq_lookup"]
        metrics = [45.0, 2.0, 0.8]

        res = self.engine.choice(context=context, options=options, metrics=metrics)

        self.assertIsInstance(res, DecisionResult)
        self.assertEqual(res.task, "choice")
        self.assertIn(res.value, options)
        self.assertIsInstance(res.probabilities, dict)
        self.assertEqual(len(res.probabilities), len(options))
        # Probabilities must sum to ~1.0
        self.assertAlmostEqual(sum(res.probabilities.values()), 1.0, places=4)
        self.assertEqual(res.state.shape, (64,))

    def test_score_primitive(self):
        """Test continuous escalation/frustration scoring primitive (1.0 to 10.0)."""
        context = "Customer has waited 45 minutes and asked for manager 3 times."
        rubric = "Rate customer churn risk from 1.0 (calm) to 10.0 (extreme churn risk)."

        res = self.engine.score(context=context, rubric=rubric)

        self.assertIsInstance(res, DecisionResult)
        self.assertEqual(res.task, "score")
        self.assertGreaterEqual(res.value, 1.0)
        self.assertLessEqual(res.value, 10.0)
        self.assertEqual(res.state.shape, (64,))

    def test_recurrent_state_persistence(self):
        """
        Verify that passing recurrent working memory (prev_state -> next_state)
        accumulates across multi-turn sequential calls.
        """
        turn1_context = "Turn 1: Hi, I need help with my account."
        res1 = self.engine.score(context=turn1_context, rubric="Frustration score 1-10")
        state_t1 = res1.state

        turn2_context = "Turn 2: Why is this taking so long? Nobody answered my ticket!"
        res2 = self.engine.score(context=turn2_context, rubric="Frustration score 1-10", state=state_t1)
        state_t2 = res2.state

        turn3_context = "Turn 3: Cancel my entire subscription immediately!"
        res3 = self.engine.score(context=turn3_context, rubric="Frustration score 1-10", state=state_t2)
        state_t3 = res3.state

        # State vectors must evolve across turns and not remain identical
        self.assertFalse(np.allclose(state_t1, state_t2))
        self.assertFalse(np.allclose(state_t2, state_t3))
        self.assertEqual(state_t3.shape, (64,))


class TestHybridRouter(unittest.TestCase):
    def setUp(self):
        self.engine = SystemOneEngine(mock_mode=True)
        self.router = HybridSystemRouter(
            engine=self.engine,
            uncertainty_threshold=0.60,
            ambiguity_margin=0.08,
        )

    def test_system1_confident_routing(self):
        """Test that in-distribution queries with low uncertainty stay in System 1."""
        # Force low uncertainty mock output
        decision = self.router.route_noul(
            context="Standard search query",
            question="Is this benign?"
        )
        self.assertIsInstance(decision, RoutingDecision)
        self.assertIn(decision.engine_used, ["system1", "system2"])
        self.assertGreater(decision.total_latency_ms, 0.0)

    def test_system2_escalation_on_high_uncertainty(self):
        """Test that high epistemic uncertainty forces System 2 LLM fallback."""
        # Create a router with very low uncertainty threshold to force escalation
        sensitive_router = HybridSystemRouter(
            engine=self.engine,
            uncertainty_threshold=0.01, # Any uncertainty triggers escalation
        )
        decision = sensitive_router.route_noul(
            context="Corrupted adversarial query \x00\xff bypass matrix",
            question="Evaluate validity"
        )
        self.assertTrue(decision.escalated)
        self.assertEqual(decision.engine_used, "system2")
        self.assertIn("High epistemic uncertainty", decision.escalation_reason)
        self.assertEqual(decision.estimated_cost_multiplier, 50.0)


class TestCalibrationMetrics(unittest.TestCase):
    def test_perfect_calibration_ece(self):
        """Test that perfectly calibrated predictions produce near-zero ECE."""
        # Bin 1: 100 samples with confidence 0.80, 80 correct (acc = 0.80) -> diff = 0.0
        # Bin 2: 100 samples with confidence 0.20, 20 correct (acc = 0.20) -> diff = 0.0
        confidences = [0.80] * 100 + [0.20] * 100
        predictions = [1] * 200
        # First 100: 80 true 1s, 20 true 0s (acc = 0.80)
        # Second 100: 20 true 1s, 80 true 0s (acc = 0.20)
        ground_truth = ([1] * 80 + [0] * 20) + ([1] * 20 + [0] * 80)

        ece_res = compute_ece(confidences, predictions, ground_truth, num_bins=10)
        self.assertLess(ece_res["ece"], 0.01)
        self.assertLessEqual(ece_res["mce"], 0.01)

    def test_uncalibrated_ece(self):
        """Test that overconfident wrong predictions produce high ECE."""
        confidences = [0.99] * 100
        predictions = [1] * 100
        ground_truth = [0] * 100 # 0% accuracy despite 99% confidence

        ece_res = compute_ece(confidences, predictions, ground_truth, num_bins=10)
        self.assertGreater(ece_res["ece"], 0.90)

    def test_ascii_reliability_diagram_renders(self):
        """Test that ASCII calibration diagram formats properly without error."""
        confidences = np.random.uniform(0.1, 0.9, 200)
        predictions = np.random.randint(0, 2, 200)
        ground_truth = np.random.randint(0, 2, 200)

        ece_res = compute_ece(confidences, predictions, ground_truth, num_bins=5)
        diagram = ascii_reliability_diagram(ece_res)
        self.assertIn("RELIABILITY DIAGRAM", diagram)
        self.assertIn("Bin Range", diagram)


if __name__ == "__main__":
    unittest.main()
