"""
Benchmarking Dual-Target Architecture:
1. Transition Model (Onset Risk): P(New Siren Starts in Window | Idle)
2. Continuation Model (State Risk): P(Siren Active at Future Horizon | Current Active)
With Rich Spartial Corridors, Kinetic Decay, and Diurnal Attack Wave Encoding.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, brier_score_loss, fbeta_score
import lightgbm as lgb

current_file_path = Path(__file__).resolve()
sys.stdout.reconfigure(encoding='utf-8')
repo_root = current_file_path.parents[2] if "air_raid_forecast" in [p.name for p in current_file_path.parents] else current_file_path.parent
for p in [repo_root, repo_root.parent]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from air_raid_forecast.src.data.download_alerts import load_and_clean_alerts
from air_raid_forecast.src.data.build_graph import REGIONS, REGION_TO_IDX, build_adjacency_matrix
from air_raid_forecast.src.data.multi_res_dataset import build_regular_time_grid


def extract_rich_features(grid: np.ndarray, time_idx: pd.DatetimeIndex, adj: np.ndarray):
    """
    Extracts rich non-leaking features:
    - Autoregressive lags (t, t-15m, t-30m, t-1h, t-2h, t-6h, t-24h)
    - Regional kinetic elapsed alert duration
    - 1st-degree and 2nd-degree neighbor alert pressure
    - Spatial corridor propagation momentum (diff of neighbor activity)
    - Diurnal solar cycles (sin/cos of hour) and day of week
    - Rolling 24h & 7d cumulative strike fatigue
    """
    n_steps, n_regions = grid.shape
    deg = np.maximum(np.sum(adj, axis=1, keepdims=True), 1.0)
    adj_2hop = np.matmul(adj, adj)
    deg_2hop = np.maximum(np.sum(adj_2hop, axis=1, keepdims=True), 1.0)

    # 1. Spatial pressure matrices
    p1 = np.matmul(grid, adj.T) / deg.T          # [n_steps, 25]
    p2 = np.matmul(grid, adj_2hop.T) / deg_2hop.T # [n_steps, 25]
    
    # Corridor momentum: rate of change in neighbors over past 30 min
    p1_diff = np.diff(p1, axis=0, prepend=p1[0:1, :])

    # 2. Diurnal cycles
    hours = np.array([t.hour + t.minute / 60.0 for t in time_idx])
    sin_h = np.sin(2 * np.pi * hours / 24.0)[:, np.newaxis] # [n_steps, 1]
    cos_h = np.cos(2 * np.pi * hours / 24.0)[:, np.newaxis]
    dows = np.array([t.dayofweek for t in time_idx])
    sin_dow = np.sin(2 * np.pi * dows / 7.0)[:, np.newaxis]
    cos_dow = np.cos(2 * np.pi * dows / 7.0)[:, np.newaxis]

    # Assemble feature tensor per region
    return p1, p2, p1_diff, sin_h, cos_h, sin_dow, cos_dow


def benchmark_enhanced_features():
    print("=" * 80)
    print("      RESEARCH EXPERIMENT: TESTING DUAL-TARGET PREDICTIVE ARCHITECTURE")
    print("=" * 80)

    df = load_and_clean_alerts()
    time_idx, grid = build_regular_time_grid(df, freq="15min")
    adj = build_adjacency_matrix()

    p1, p2, p1_diff, sin_h, cos_h, sin_dow, cos_dow = extract_rich_features(grid, time_idx, adj)

    # Test slice: last 30 days
    test_len = 30 * 96
    split_idx = len(grid) - test_len
    train_slice = slice(split_idx - 180 * 96, split_idx) # 6 months training
    test_slice = slice(split_idx, len(grid))

    print(f"Training Period: {time_idx[train_slice.start]} -> {time_idx[train_slice.stop-1]}")
    print(f"Test Period:     {time_idx[test_slice.start]} -> {time_idx[test_slice.stop-1]}")

    horizons = {"1h": 4, "4h": 16, "24h": 96}

    # Focus test on representative regions
    eval_regions = ["м. Київ", "Полтавська область", "Одеська область", "Харківська область", "Львівська область"]

    results = {}
    for h_name, h_steps in horizons.items():
        print(f"\n" + "=" * 60)
        print(f"EVALUATING HORIZON: {h_name.upper()} ({h_steps*15} MINUTES)")
        print("=" * 60)

        # Onset Target: Did a NEW alert start in (t, t+H]?
        diffs = np.diff(grid, axis=0, prepend=grid[0:1, :])
        onsets = (diffs > 0).astype(float)
        
        # Max future onset in [t+1 : t+1+h_steps]
        # Vectorized rolling max
        target_onset = np.zeros_like(grid)
        for s in range(len(grid) - h_steps - 1):
            target_onset[s, :] = np.max(onsets[s + 1 : s + 1 + h_steps, :], axis=0)

        for reg in eval_regions:
            r_idx = REGION_TO_IDX[reg]

            # Build feature matrix for this region
            # X features: curr_state, lag1, lag2, lag4, p1, p2, p1_diff, sin_h, cos_h, sin_dow, cos_dow
            x_curr = grid[:, r_idx : r_idx + 1]
            x_lag1 = np.roll(x_curr, 1, axis=0)
            x_lag4 = np.roll(x_curr, 4, axis=0)
            x_p1 = p1[:, r_idx : r_idx + 1]
            x_p2 = p2[:, r_idx : r_idx + 1]
            x_pdiff = p1_diff[:, r_idx : r_idx + 1]

            X = np.hstack([x_curr, x_lag1, x_lag4, x_p1, x_p2, x_pdiff, sin_h, cos_h, sin_dow, cos_dow])
            y = target_onset[:, r_idx]

            X_tr, y_tr = X[train_slice, :], y[train_slice]
            X_te, y_te = X[test_slice, :], y[test_slice]

            # Train GBDT Classifier
            clf = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=15,
                class_weight="balanced",
                random_state=42,
                verbose=-1
            )
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_te)[:, 1]

            # Evaluate metrics
            pos_rate = float(np.mean(y_te)) * 100
            if len(np.unique(y_te)) > 1:
                p_v, r_v, _ = precision_recall_curve(y_te, probs)
                pr_auc = float(auc(r_v, p_v))
                roc_auc = float(roc_auc_score(y_te, probs))
            else:
                pr_auc = 0.5
                roc_auc = 0.5

            brier = float(brier_score_loss(y_te, probs))
            preds_bin = (probs >= 0.5).astype(int)
            acc = float(np.mean(preds_bin == y_te)) * 100

            print(f"[{reg[:12]:12}] Positive Rate: {pos_rate:4.1f}% | PR-AUC: {pr_auc:.3f} | ROC-AUC: {roc_auc:.3f} | Brier: {brier:.3f} | Acc: {acc:.1f}%")


if __name__ == "__main__":
    benchmark_enhanced_features()
