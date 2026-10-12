"""
Stand-Alone Verification Script for the Overfitting Diagnostics Engine.
Verifies metrics computation, persistence benchmarking, and diagnostic thresholds
using synthetic test tensors.
"""

import sys
from pathlib import Path
import unittest
import numpy as np

# Ensure project root and parent are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from air_raid_forecast.src.training.evaluate_diagnostics import (
    compute_expected_calibration_error,
    evaluate_horizon_metrics,
    compute_persistence_baseline,
    diagnose_overfitting,
)


class TestDiagnosticsAndOverfitting(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_samples = 100
        self.n_regions = 25
        self.y_true = np.random.choice([0.0, 1.0], size=(self.n_samples, self.n_regions, 3), p=[0.7, 0.3])
        self.y_prob = np.random.uniform(0.0, 1.0, size=(self.n_samples, self.n_regions, 3))
        self.x_fast = np.zeros((self.n_samples, self.n_regions, 8, 2))
        self.x_fast[:, :, -1, 0] = self.y_true[:, :, 0] # perfect persistence on 1h

    def test_ece_computation(self):
        ece = compute_expected_calibration_error(self.y_true[:, :, 0], self.y_prob[:, :, 0], n_bins=5)
        self.assertIsInstance(ece, float)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_horizon_metrics(self):
        metrics = evaluate_horizon_metrics(self.y_true[:, :, 0], self.y_prob[:, :, 0], threshold=0.5)
        expected_keys = ["pr_auc", "roc_auc", "brier_score", "ece", "f2_score", "precision", "recall"]
        for k in expected_keys:
            self.assertIn(k, metrics)

    def test_persistence_baseline(self):
        pers = compute_persistence_baseline(self.x_fast, self.y_true)
        self.assertIn("1h", pers)
        self.assertIn("4h", pers)
        self.assertIn("24h", pers)

    def test_overfitting_verdicts(self):
        # Case 1: Healthy generalization
        diag_healthy = diagnose_overfitting(
            train_loss=0.40,
            val_loss=0.45,
            model_pr_auc={"1h": 0.75, "4h": 0.65, "24h": 0.55},
            baseline_pr_auc={"1h": 0.60, "4h": 0.50, "24h": 0.40}
        )
        self.assertEqual(diag_healthy["status"], "HEALTHY_GENERALIZATION")

        # Case 2: Divergence / Overfitting
        diag_overfit = diagnose_overfitting(
            train_loss=0.20,
            val_loss=0.40, # Ratio = 2.0
            model_pr_auc={"1h": 0.65, "4h": 0.50, "24h": 0.40},
            baseline_pr_auc={"1h": 0.60, "4h": 0.50, "24h": 0.40}
        )
        self.assertEqual(diag_overfit["status"], "OVERFITTED")


if __name__ == "__main__":
    unittest.main()
