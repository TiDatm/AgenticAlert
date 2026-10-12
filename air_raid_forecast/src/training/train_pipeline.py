"""
Production Training Pipeline with Early Stopping, Regularization,
and Automated Overfitting Detection.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
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
from air_raid_forecast.src.training.evaluate_diagnostics import (
    evaluate_horizon_metrics,
    compute_persistence_baseline,
    diagnose_overfitting,
)


class PyTorchGridDataset(Dataset):
    def __init__(self, mr_dataset: MultiResolutionDataset, sample_stride: int = 4):
        self.mr = mr_dataset
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


def train_with_overfitting_checks(
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 3,
    sample_stride: int = 4,   # Sample every hour
    val_days: int = 30,       # 30 days validation
    test_days: int = 30,      # Latest 30 days holdout
    save_dir: str = "checkpoints"
) -> str:
    """
    Executes regularized training, tracks generalization gap across epochs,
    applies early stopping, and generates an overfitting diagnostic report.
    """
    if not os.path.isabs(save_dir):
        save_dir = str(_PROJECT_ROOT / save_dir)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Training Pipeline on: {device}")

    # 1. Load alerts data and build time grid
    df = load_and_clean_alerts()
    time_idx, grid = build_regular_time_grid(df, freq="15min")
    adj = build_adjacency_matrix()
    lap = get_normalized_laplacian(adj)
    lap_t = torch.from_numpy(lap).float().to(device)

    # 2. Sequential Temporal Splits (Train -> Val -> Test)
    steps_per_day = 24 * 4 # 96 steps
    test_steps = test_days * steps_per_day
    val_steps = val_days * steps_per_day

    total_len = len(time_idx)
    test_start = total_len - test_steps
    val_start = test_start - val_steps

    print("=" * 70)
    print("CHRONOLOGICAL TEMPORAL SPLITS (ZERO DATA LEAKAGE)")
    print(f"  Training Period:   {time_idx.min()} -> {time_idx[val_start]}")
    print(f"  Validation Period: {time_idx[val_start]} -> {time_idx[test_start]}")
    print(f"  Holdout Test:      {time_idx[test_start]} -> {time_idx.max()}")
    print("=" * 70)

    # 3. Create Datasets
    train_mr = MultiResolutionDataset(grid[:val_start, :], time_idx[:val_start], adj)
    val_mr = MultiResolutionDataset(grid[val_start - 96 : test_start, :], time_idx[val_start - 96 : test_start], adj)
    test_mr = MultiResolutionDataset(grid[test_start - 96 :, :], time_idx[test_start - 96 :], adj)

    train_loader = DataLoader(PyTorchGridDataset(train_mr, sample_stride), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PyTorchGridDataset(val_mr, sample_stride), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(PyTorchGridDataset(test_mr, sample_stride), batch_size=batch_size, shuffle=False)

    # 4. Model, Criterion, Optimizer
    model = MultiResSTGNN(num_nodes=25).to(device)
    criterion = MultiTaskFocalLoss(gamma=2.0, alpha=0.75)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    from tqdm import tqdm

    # Tracking variables for overfitting detection
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "loss_gap": []}
    best_weights_path = os.path.join(save_dir, "best_stgnn_model.pt")

    # 5. Training Epochs with Generalization Tracking
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
        for x_f, x_s, y in train_pbar:
            x_f, x_s, y = x_f.to(device), x_s.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x_f, x_s, lap_t)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_train += 1

            # Live loss indicator in progress bar
            running_loss = train_loss / n_train
            train_pbar.set_postfix(loss=f"{running_loss:.4f}")

        avg_train = train_loss / max(n_train, 1)

        # Validation pass
        model.eval()
        val_loss = 0.0
        n_val = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]  ", leave=False)
        with torch.no_grad():
            for x_f, x_s, y in val_pbar:
                x_f, x_s, y = x_f.to(device), x_s.to(device), y.to(device)
                logits = model(x_f, x_s, lap_t)
                loss = criterion(logits, y)
                val_loss += loss.item()
                n_val += 1
                val_pbar.set_postfix(val_loss=f"{val_loss / n_val:.4f}")

        avg_val = val_loss / max(n_val, 1)
        loss_ratio = avg_val / max(avg_train, 1e-6)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["loss_gap"].append(round(loss_ratio, 4))

        status_flag = "OK" if loss_ratio <= 1.25 else ("WARN: Divergence" if loss_ratio <= 1.45 else "ALERT: Overfitting")
        print(f"Epoch [{epoch:02d}/{epochs:02d}] -- Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Gen Ratio: {loss_ratio:.2f} [{status_flag}]")

        # Early stopping logic
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), best_weights_path)
            print(f"  [+] Checkpoint improved ({avg_val:.4f}) -> Saved {best_weights_path}")
        else:
            patience_counter += 1
            print(f"  [-] Val loss did not improve ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} to prevent overfitting.")
                break

    # 6. Evaluation & Overfitting Diagnosis on Holdout Test Set
    print("\n" + "=" * 70)
    print("COMPREHENSIVE HOLDOUT EVALUATION & OVERFITTING VERIFICATION")
    print("=" * 70)

    model.load_state_dict(torch.load(best_weights_path))
    model.eval()

    all_preds = []
    all_targets = []
    all_x_fast = []

    test_pbar = tqdm(test_loader, desc="Evaluating Holdout Test Set", leave=True)
    with torch.no_grad():
        for x_f, x_s, y in test_pbar:
            x_f_dev, x_s_dev = x_f.to(device), x_s.to(device)
            logits = model(x_f_dev, x_s_dev, lap_t)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(y.numpy())
            all_x_fast.append(x_f.numpy())

    preds = np.concatenate(all_preds, axis=0)     # [N, 25, 3]
    targets = np.concatenate(all_targets, axis=0) # [N, 25, 3]
    x_fast_all = np.concatenate(all_x_fast, axis=0)

    # Compute persistence baselines
    pers_metrics = compute_persistence_baseline(x_fast_all, targets)

    horizons = ["1h", "4h", "24h"]
    h_names = ["Next 1 Hour (Tactical)", "Next 4 Hours (Operational)", "Next 24 Hours (Strategic)"]
    model_metrics = {}
    model_pr_aucs = {}
    base_pr_aucs = {}

    for idx, (h_code, h_label) in enumerate(zip(horizons, h_names)):
        p_h = preds[:, :, idx]
        t_h = targets[:, :, idx]

        m_res = evaluate_horizon_metrics(t_h, p_h, threshold=0.5)
        model_metrics[h_code] = m_res
        model_pr_aucs[h_code] = m_res["pr_auc"]
        base_pr_aucs[h_code] = pers_metrics[h_code]["pr_auc"]

        print(f"\n--- Horizon: {h_label} ---")
        print(f"  Model PR-AUC:       {m_res['pr_auc']:.4f}  (Baseline Persistence: {pers_metrics[h_code]['pr_auc']:.4f})")
        print(f"  Model ROC-AUC:      {m_res['roc_auc']:.4f}")
        print(f"  Model Brier Score:  {m_res['brier_score']:.4f}")
        print(f"  Model ECE (Calib):  {m_res['ece']:.4f}")
        print(f"  Model F2 Utility:   {m_res['f2_score']:.4f}")

    # Automated overfitting diagnosis
    diagnosis = diagnose_overfitting(
        train_loss=history["train_loss"][-1],
        val_loss=history["val_loss"][-1],
        model_pr_auc=model_pr_aucs,
        baseline_pr_auc=base_pr_aucs
    )

    print("\n" + "=" * 70)
    print("OVERFITTING & GENERALIZATION DIAGNOSTIC REPORT")
    print(f"  Verdict:           {diagnosis['status']}")
    print(f"  Loss Ratio:        {diagnosis['generalization_loss_ratio']}")
    print(f"  PR-AUC Lifts:      {diagnosis['baseline_pr_auc_lifts']}")
    print(f"  Recommendation:    {diagnosis['recommendation']}")
    print("=" * 70)

    # Save full metrics report to disk
    report_path = os.path.join(save_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "training_history": history,
            "model_metrics": model_metrics,
            "persistence_baseline_metrics": pers_metrics,
            "overfitting_diagnosis": diagnosis
        }, f, indent=2)

    print(f"Detailed evaluation and overfitting report saved to: {report_path}")
    return report_path


if __name__ == "__main__":
    train_with_overfitting_checks()
