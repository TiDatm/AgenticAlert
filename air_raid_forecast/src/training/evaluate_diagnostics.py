"""
Comprehensive Evaluation, Benchmarking & Overfitting Diagnostic Engine.
Computes:
1. Advanced Metrics: PR-AUC, ROC-AUC, Brier Score, F2 Utility, ECE (Calibration)
2. Baseline Benchmarking: Persistence Baseline and Climatology Baseline
3. Overfitting Analysis: Generalization Loss Gap and Overfitting Severity Diagnosis
"""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, brier_score_loss, fbeta_score


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE) across binned confidence intervals.
    """
    y_true_flat = y_true.flatten()
    y_prob_flat = np.clip(y_prob.flatten(), 1e-6, 1.0 - 1e-6)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true_flat)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (y_prob_flat > bin_lower) & (y_prob_flat <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true_flat[in_bin])
            avg_confidence_in_bin = np.mean(y_prob_flat[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(round(ece, 4))


def evaluate_horizon_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes classification, calibration, and utility metrics for a single horizon.
    """
    y_true_flat = y_true.flatten()
    y_prob_flat = y_prob.flatten()

    if len(np.unique(y_true_flat)) < 2:
        return {
            "pr_auc": 0.0, "roc_auc": 0.5, "brier_score": 0.0,
            "ece": 0.0, "f2_score": 0.0, "precision": 0.0, "recall": 0.0
        }

    # PR-AUC
    p_vals, r_vals, _ = precision_recall_curve(y_true_flat, y_prob_flat)
    pr_auc = float(auc(r_vals, p_vals))

    # ROC-AUC
    roc_auc = float(roc_auc_score(y_true_flat, y_prob_flat))

    # Brier Score & ECE
    brier = float(brier_score_loss(y_true_flat, y_prob_flat))
    ece = compute_expected_calibration_error(y_true_flat, y_prob_flat)

    # Cost-Sensitive Utility (F2)
    y_pred_bin = (y_prob_flat >= threshold).astype(int)
    f2 = float(fbeta_score(y_true_flat, y_pred_bin, beta=2, zero_division=0))

    tp = np.sum((y_pred_bin == 1) & (y_true_flat == 1))
    fp = np.sum((y_pred_bin == 1) & (y_true_flat == 0))
    fn = np.sum((y_pred_bin == 0) & (y_true_flat == 1))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "f2_score": round(f2, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def compute_persistence_baseline(
    x_fast: np.ndarray,      # [N_samples, 25, 8, F]
    y_true: np.ndarray       # [N_samples, 25, 3]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates naive spatial persistence baseline (predicting current status x_t).
    """
    # Most recent status is index -1 in fast stream, feature 0 (is_active)
    curr_state = x_fast[:, :, -1, 0] # [N_samples, 25]

    horizons = ["1h", "4h", "24h"]
    baseline_metrics = {}

    for h_idx, h_name in enumerate(horizons):
        t_h = y_true[:, :, h_idx]
        baseline_metrics[h_name] = evaluate_horizon_metrics(t_h, curr_state, threshold=0.5)

    return baseline_metrics


def diagnose_overfitting(
    train_loss: float,
    val_loss: float,
    model_pr_auc: Dict[str, float],
    baseline_pr_auc: Dict[str, float]
) -> Dict[str, Any]:
    """
    Synthesizes training vs. validation metrics into a formal Overfitting Diagnostic Report.
    """
    loss_ratio = val_loss / max(train_loss, 1e-6)
    
    lifts = {}
    for h in ["1h", "4h", "24h"]:
        m_auc = model_pr_auc.get(h, 0.0)
        b_auc = baseline_pr_auc.get(h, 0.0)
        lifts[h] = round(m_auc - b_auc, 4)

    # Overfitting verdict rules
    if loss_ratio <= 1.25 and all(l >= 0 for l in lifts.values()):
        status = "HEALTHY_GENERALIZATION"
        recommendation = "Model exhibits strong generalization and beats the persistence baseline without memorization."
    elif loss_ratio <= 1.45:
        status = "MODERATE_DIVERGENCE"
        recommendation = "Acceptable generalization. Consider modest weight decay or early stopping."
    else:
        status = "OVERFITTED"
        recommendation = "Warning: Validation loss diverged from train loss. Increase dropout, reduce GNN layers, or apply stronger regularization."

    return {
        "status": status,
        "generalization_loss_ratio": round(loss_ratio, 4),
        "train_loss": round(train_loss, 4),
        "val_loss": round(val_loss, 4),
        "baseline_pr_auc_lifts": lifts,
        "recommendation": recommendation
    }
