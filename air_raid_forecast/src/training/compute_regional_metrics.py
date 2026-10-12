"""
Regional Metric Computation Module.
Computes evaluation metrics (PR-AUC, ROC-AUC, Precision, Recall, Accuracy, F2)
individually for every single Ukrainian oblast and Kyiv City on the holdout test set.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, fbeta_score

current_file_path = Path(__file__).resolve()
repo_root = current_file_path.parents[2] if "air_raid_forecast" in [p.name for p in current_file_path.parents] else current_file_path.parent
# Ensure both repo_root and repo_root/air_raid_forecast are available
for p in [repo_root, repo_root.parent]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from air_raid_forecast.src.data.download_alerts import load_and_clean_alerts
from air_raid_forecast.src.data.build_graph import REGIONS, REGION_TO_IDX, build_adjacency_matrix, get_normalized_laplacian
from air_raid_forecast.src.data.multi_res_dataset import build_regular_time_grid, MultiResolutionDataset
from air_raid_forecast.src.models.st_gnn import MultiResSTGNN


def compute_per_region_metrics(
    checkpoint_path: str = "air_raid_forecast/checkpoints/best_stgnn_model.pt",
    test_days: int = 14,
    sample_stride: int = 12
) -> dict:
    """
    Computes per-region prediction quality across all 25 regions on the holdout period.
    """
    device = torch.device("cpu")
    df = load_and_clean_alerts()
    time_idx, grid = build_regular_time_grid(df, freq="15min")
    adj = build_adjacency_matrix()
    lap = get_normalized_laplacian(adj)
    lap_t = torch.from_numpy(lap).float().to(device)

    # Latest holdout period
    test_steps = test_days * 96
    test_start = len(time_idx) - test_steps
    test_mr = MultiResolutionDataset(grid[test_start - 96 :, :], time_idx[test_start - 96 :], adj)

    # Load model
    model = MultiResSTGNN(num_nodes=25).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    indices = list(range(0, len(test_mr), sample_stride))
    with torch.no_grad():
        for i in indices:
            x_f, x_s, y = test_mr.get_sample(i)
            x_f_t = torch.from_numpy(x_f).unsqueeze(0).float()
            x_s_t = torch.from_numpy(x_s).unsqueeze(0).float()
            logits = model(x_f_t, x_s_t, lap_t)
            probs = torch.sigmoid(logits).squeeze(0).numpy() # [25, 3]
            all_preds.append(probs)
            all_targets.append(y)

    preds = np.stack(all_preds, axis=0)     # [N, 25, 3]
    targets = np.stack(all_targets, axis=0) # [N, 25, 3]

    regional_results = {}
    for r_idx, r_name in enumerate(REGIONS):
        regional_results[r_name] = {}
        for h_idx, h_name in enumerate(["1h", "4h", "24h"]):
            y_t = targets[:, r_idx, h_idx]
            y_p = preds[:, r_idx, h_idx]

            # Binary accuracy at 0.5 threshold
            y_pred_bin = (y_p >= 0.5).astype(int)
            acc = float(np.mean(y_pred_bin == y_t.astype(int)))

            # Confusion matrix components
            tp = np.sum((y_pred_bin == 1) & (y_t == 1))
            fp = np.sum((y_pred_bin == 1) & (y_t == 0))
            fn = np.sum((y_pred_bin == 0) & (y_t == 1))
            tn = np.sum((y_pred_bin == 0) & (y_t == 0))

            sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.5
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.5
            # Balanced Accuracy prevents 100% saturation on degenerate 0% or 100% positive rate regions
            balanced_acc = float((sens + spec) / 2.0)

            # PR-AUC & ROC-AUC
            if len(np.unique(y_t)) > 1:
                p_v, r_v, _ = precision_recall_curve(y_t, y_p)
                pr_auc = float(auc(r_v, p_v))
                roc_auc = float(roc_auc_score(y_t, y_p))
            else:
                pr_auc = 0.5
                roc_auc = 0.5

            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f2 = float(fbeta_score(y_t, y_pred_bin, beta=2, zero_division=0))

            regional_results[r_name][h_name] = {
                "accuracy": round(acc * 100, 1),
                "balanced_accuracy": round(balanced_acc * 100, 1),
                "pr_auc": round(pr_auc, 3),
                "roc_auc": round(roc_auc, 3),
                "precision": round(prec * 100, 1),
                "recall": round(rec * 100, 1),
                "f2_score": round(f2, 3),
                "positive_rate": round(float(np.mean(y_t)) * 100, 1)
            }

    # Save to disk
    out_file = os.path.join(str(repo_root), "air_raid_forecast", "checkpoints", "regional_accuracy.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(regional_results, f, ensure_ascii=False, indent=2)

    return regional_results


if __name__ == "__main__":
    res = compute_per_region_metrics()
    print("Computed regional accuracy metrics for", len(res), "regions.")
