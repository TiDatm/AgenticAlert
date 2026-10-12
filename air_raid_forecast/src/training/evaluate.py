"""
Evaluation Metrics & Calibration for Air Raid Alert Forecasting.
Computes PR-AUC, ROC-AUC, Brier score, and F2 cost-sensitive utility.
"""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, brier_score_loss, fbeta_score


def evaluate_predictions(
    y_true: np.ndarray,      # [N_samples, 25] binary ground truth
    y_prob: np.ndarray,      # [N_samples, 25] calibrated probabilities in [0, 1]
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes comprehensive evaluation metrics across all samples.
    """
    y_true_flat = y_true.flatten()
    y_prob_flat = y_prob.flatten()

    # Avoid crash if all 0 or all 1
    if len(np.unique(y_true_flat)) < 2:
        return {
            "pr_auc": 0.0,
            "roc_auc": 0.5,
            "brier_score": float(np.mean((y_prob_flat - y_true_flat) ** 2)),
            "f2_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    # 1. PR-AUC (Primary Metric for Imbalanced Alerts)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true_flat, y_prob_flat)
    pr_auc = float(auc(recall_vals, precision_vals))

    # 2. ROC-AUC
    roc_auc = float(roc_auc_score(y_true_flat, y_prob_flat))

    # 3. Brier Score (Probability Calibration)
    brier = float(brier_score_loss(y_true_flat, y_prob_flat))

    # 4. F2 Score (Asymmetric False Negative Penalty)
    y_pred_bin = (y_prob_flat >= threshold).astype(int)
    f2 = float(fbeta_score(y_true_flat, y_pred_bin, beta=2, zero_division=0))

    # Precision & Recall
    tp = np.sum((y_pred_bin == 1) & (y_true_flat == 1))
    fp = np.sum((y_pred_bin == 1) & (y_true_flat == 0))
    fn = np.sum((y_pred_bin == 0) & (y_true_flat == 1))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "brier_score": round(brier, 4),
        "f2_score": round(f2, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = 2.0
) -> float:
    """
    Finds probability threshold maximizing F_beta utility on validation set.
    """
    best_th = 0.5
    best_f = -1.0
    y_true_flat = y_true.flatten()
    y_prob_flat = y_prob.flatten()

    for th in np.linspace(0.1, 0.9, 17):
        preds = (y_prob_flat >= th).astype(int)
        score = fbeta_score(y_true_flat, preds, beta=beta, zero_division=0)
        if score > best_f:
            best_f = score
            best_th = float(th)

    return best_th
