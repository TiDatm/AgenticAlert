"""
Automated Anti-Leakage & Offline-Online Parity Unit Tests.
Verifies that feature tensors generated at time t do not change regardless
of whether future events occur or are injected.
"""

import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch

# Ensure project root and parent are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from air_raid_forecast.src.data.build_graph import build_adjacency_matrix, get_normalized_laplacian, REGIONS
from air_raid_forecast.src.data.multi_res_dataset import MultiResolutionDataset
from air_raid_forecast.src.models.st_gnn import MultiResSTGNN


class TestAntiLeakageAndParity(unittest.TestCase):
    def setUp(self):
        self.adj = build_adjacency_matrix()
        self.lap = get_normalized_laplacian(self.adj)
        self.n_regions = len(REGIONS)
        self.n_steps = 200

        # Construct synthetic base grid
        np.random.seed(42)
        self.grid_base = np.random.choice([0.0, 1.0], size=(self.n_steps, self.n_regions), p=[0.8, 0.2]).astype(np.float32)
        self.times = pd.date_range("2024-01-01", periods=self.n_steps, freq="15min", tz="UTC")

    def test_strict_temporal_causality(self):
        """
        Verify that features at forecast step t do not change if we modify future events at t + 1 .. t + 50.
        """
        eval_idx = 10
        ds1 = MultiResolutionDataset(self.grid_base, self.times, self.adj)
        x_fast1, x_slow1, y1 = ds1.get_sample(eval_idx)

        # Modify future events strictly AFTER step t
        t_sample = ds1.min_idx + eval_idx
        grid_perturbed = self.grid_base.copy()
        grid_perturbed[t_sample + 1 :, :] = 1.0 - grid_perturbed[t_sample + 1 :, :]

        ds2 = MultiResolutionDataset(grid_perturbed, self.times, self.adj)
        x_fast2, x_slow2, y2 = ds2.get_sample(eval_idx)

        # Bit-for-bit assertion: Past and instantaneous features MUST NOT change
        np.testing.assert_array_equal(
            x_fast1, x_fast2,
            err_msg="LEAKAGE DETECTED: Fast stream features changed when future values were altered!"
        )
        np.testing.assert_array_equal(
            x_slow1, x_slow2,
            err_msg="LEAKAGE DETECTED: Slow stream features changed when future values were altered!"
        )

    def test_stgnn_output_dimensions(self):
        """
        Verifies that model accepts multi-resolution inputs and outputs shape [B, 25, 3].
        """
        model = MultiResSTGNN(num_nodes=self.n_regions)
        b = 2
        x_f = torch.randn(b, self.n_regions, 8, 2)
        x_s = torch.randn(b, self.n_regions, 24, 3)
        lap_t = torch.from_numpy(self.lap).float()

        out = model(x_f, x_s, lap_t)
        self.assertEqual(out.shape, (b, self.n_regions, 3))


if __name__ == "__main__":
    unittest.main()
