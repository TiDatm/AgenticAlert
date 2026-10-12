"""
Experimental Test: Multi-Horizon Onset Transition Modeling vs Persistence.
Evaluates the hypothesis that predicting *NEW ALERT ONSETS* resolves
the 100% saturation in frontline oblasts and 100% trivial negative in western oblasts.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, brier_score_loss, fbeta_score

current_file_path = Path(__file__).resolve()
repo_root = current_file_path.parents[2] if "air_raid_forecast" in [p.name for p in current_file_path.parents] else current_file_path.parent
for p in [repo_root, repo_root.parent]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from air_raid_forecast.src.data.download_alerts import load_and_clean_alerts
from air_raid_forecast.src.data.build_graph import REGIONS, REGION_TO_IDX, build_adjacency_matrix
from air_raid_forecast.src.data.multi_res_dataset import build_regular_time_grid


def run_onset_transition_experiment(test_days: int = 14):
    print("=" * 70)
    print("HYPOTHESIS TEST: NEW ALERT ONSET TARGET FORMULATION")
    print("=" * 70)

    df = load_and_clean_alerts()
    time_idx, grid = build_regular_time_grid(df, freq="15min")
    adj = build_adjacency_matrix()
    deg = np.maximum(np.sum(adj, axis=1, keepdims=True), 1.0)

    # Latest test slice
    test_len = test_days * 96
    test_grid = grid[-test_len:, :]
    test_times = time_idx[-test_len:]

    print(f"Test period: {test_times.min()} to {test_times.max()} ({len(test_times)} steps)")

    # 1. State targets: "Is alert active anywhere in (t, t+H]?"
    # 2. Onset targets: "Given idle at t, does a NEW alert start in (t, t+H]?"
    
    # Calculate onsets: cell becomes 1 at step k when it was 0 at k-1
    diffs = np.diff(test_grid, axis=0, prepend=test_grid[0:1, :])
    onsets = (diffs > 0).astype(float)

    horizons = {"1h": 4, "4h": 16, "24h": 96}

    for h_name, h_steps in horizons.items():
        print(f"\n--- Horizon: {h_name} ({h_steps*15} minutes) ---")
        
        # Binary target: new onset starts in the window
        # Sliding window max over future onsets
        target_onset = np.zeros((len(test_grid) - h_steps, 25))
        for step in range(len(test_grid) - h_steps):
            target_onset[step, :] = np.max(onsets[step + 1 : step + 1 + h_steps, :], axis=0)

        # Baseline: Current state (persistence)
        curr_state = test_grid[: len(test_grid) - h_steps, :]
        # Neighbor corridor pressure feature
        neighbor_pressure = np.matmul(curr_state, adj.T) / deg.T

        print("Regional Positive Rates for NEW ALERT ONSET (Kharkiv vs Lviv vs Kyiv):")
        kh_idx = REGION_TO_IDX["Харківська область"]
        lv_idx = REGION_TO_IDX["Львівська область"]
        ky_idx = REGION_TO_IDX["м. Київ"]

        print(f"  Kharkiv Onset Target Positive Rate: {np.mean(target_onset[:, kh_idx])*100:.1f}%")
        print(f"  Kyiv City Onset Target Positive Rate: {np.mean(target_onset[:, ky_idx])*100:.1f}%")
        print(f"  Lviv Onset Target Positive Rate:      {np.mean(target_onset[:, lv_idx])*100:.1f}%")
        print(f"  National Mean Onset Rate:             {np.mean(target_onset)*100:.1f}%")

        # Evaluate neighbor pressure as predictor of new onsets
        kh_auc = roc_auc_score(target_onset[:, kh_idx], neighbor_pressure[:, kh_idx]) if len(np.unique(target_onset[:, kh_idx])) > 1 else 0.5
        ky_auc = roc_auc_score(target_onset[:, ky_idx], neighbor_pressure[:, ky_idx]) if len(np.unique(target_onset[:, ky_idx])) > 1 else 0.5
        print(f"  Neighbor Corridor ROC-AUC for Kyiv onsets:    {ky_auc:.3f}")
        print(f"  Neighbor Corridor ROC-AUC for Kharkiv onsets: {kh_auc:.3f}")


if __name__ == "__main__":
    run_onset_transition_experiment()
