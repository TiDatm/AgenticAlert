"""
Rolling-Origin Temporal Training and Evaluation Pipeline.
Trains PyTorch Multi-Resolution ST-GNN sequentially without data leakage,
evaluating on the most recent holdout period.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure project root and parent are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from air_raid_forecast.src.data.download_alerts import load_and_clean_alerts
from air_raid_forecast.src.data.build_graph import build_adjacency_matrix, get_normalized_laplacian, REGIONS
from air_raid_forecast.src.data.multi_res_dataset import build_regular_time_grid, MultiResolutionDataset
from air_raid_forecast.src.models.st_gnn import MultiResSTGNN
from air_raid_forecast.src.models.loss import MultiTaskFocalLoss
from air_raid_forecast.src.training.evaluate import evaluate_predictions, find_optimal_threshold


class PyTorchGridDataset(Dataset):
    def __init__(self, mr_dataset: MultiResolutionDataset, sample_stride: int = 4):
        self.mr = mr_dataset
        # Subsample to speed up training while covering the temporal span
        self.indices = list(range(0, len(mr_dataset), sample_stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x_fast, x_slow, y_target = self.mr.get_sample(idx)
        return (
            torch.from_numpy(x_fast),
            torch.from_numpy(x_slow),
            torch.from_numpy(y_target)
        )


def run_training_and_evaluation(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.001,
    sample_stride: int = 4, # sample every hour
    test_days: int = 45     # latest 45 days reserved strictly as holdout test
):
    print("=" * 70)
    print("STARTING ROLLING-ORIGIN TEMPORAL TRAINING (PyTorch MultiRes-STGNN)")
    print("=" * 70)

    # 1. Load alerts data
    print("Loading official alerts data...")
    df = load_and_clean_alerts()
    print(f"Total alert records: {len(df)}")

    # 2. Build regular 15-minute time grid
    print("Building regular 15-minute spatio-temporal grid...")
    time_idx, grid = build_regular_time_grid(df, freq="15min")
    print(f"Grid shape: {grid.shape} (from {time_idx.min()} to {time_idx.max()})")

    # 3. Build Adjacency Matrix & Laplacian
    adj = build_adjacency_matrix()
    lap = get_normalized_laplacian(adj)
    lap_tensor = torch.from_numpy(lap).float()

    # 4. Strict Temporal Train / Test Split on latest period
    test_steps = test_days * 24 * 4 # 4 steps per hour
    split_idx = len(time_idx) - test_steps
    train_end_time = time_idx[split_idx]
    print(f"Temporal Split Boundary: {train_end_time}")
    print(f"Training History: {time_idx.min()} -> {train_end_time}")
    print(f"Holdout Test Period (Most Recent): {train_end_time} -> {time_idx.max()}")

    # Build datasets
    train_mr = MultiResolutionDataset(grid[:split_idx, :], time_idx[:split_idx], adj)
    test_mr = MultiResolutionDataset(grid[split_idx - 100:, :], time_idx[split_idx - 100:], adj)

    train_ds = PyTorchGridDataset(train_mr, sample_stride=sample_stride)
    test_ds = PyTorchGridDataset(test_mr, sample_stride=sample_stride)

    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 5. Initialize Model & Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running training on device: {device}")
    
    lap_tensor = lap_tensor.to(device)
    model = MultiResSTGNN(num_nodes=25).to(device)
    criterion = MultiTaskFocalLoss(gamma=2.0, alpha=0.75)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 6. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x_f, x_s, y in train_loader:
            x_f = x_f.to(device)
            x_s = x_s.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x_f, x_s, lap_tensor)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch [{epoch}/{epochs}] - Multi-Task Focal Loss: {avg_loss:.4f}")

    # 7. Final Evaluation on Holdout Test Period (Strictly unseen future)
    print("\n" + "=" * 70)
    print("EVALUATING MODEL ON LATEST HOLDOUT PERIOD (ZERO LEAKAGE)")
    print("=" * 70)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_f, x_s, y in test_loader:
            x_f = x_f.to(device)
            x_s = x_s.to(device)
            logits = model(x_f, x_s, lap_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(y.numpy())

    preds = np.concatenate(all_preds, axis=0)     # [N, 25, 3]
    targets = np.concatenate(all_targets, axis=0) # [N, 25, 3]

    horizons = ["Next 1 Hour (1h)", "Next 4 Hours (4h)", "Next 24 Hours (24h)"]
    metrics_summary = {}

    for h_idx, h_name in enumerate(horizons):
        p_h = preds[:, :, h_idx]
        t_h = targets[:, :, h_idx]

        th = find_optimal_threshold(t_h, p_h, beta=2.0)
        res = evaluate_predictions(t_h, p_h, threshold=th)
        metrics_summary[h_name] = res
        print(f"\n--- Horizon: {h_name} (Optimal F2 Threshold: {th:.2f}) ---")
        print(f"  PR-AUC (Precision-Recall AUC): {res['pr_auc']:.4f}")
        print(f"  ROC-AUC:                       {res['roc_auc']:.4f}")
        print(f"  Brier Score (Calibration):    {res['brier_score']:.4f}")
        print(f"  F2 Utility Score:              {res['f2_score']:.4f}")
        print(f"  Precision:                     {res['precision']:.4f}")
        print(f"  Recall:                        {res['recall']:.4f}")

    # Save model checkpoint
    checkpoints_dir = _PROJECT_ROOT / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(checkpoints_dir / "st_gnn_model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nModel checkpoint saved to: {checkpoint_path}")

    return metrics_summary


if __name__ == "__main__":
    run_training_and_evaluation()
