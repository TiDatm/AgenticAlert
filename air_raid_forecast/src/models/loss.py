"""
Multi-Task Cost-Sensitive Focal Loss for Air Raid Alert Forecasting.
Heavily penalizes False Negatives (missed alerts) to reflect defense utility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskFocalLoss(nn.Module):
    """
    Computes Asymmetric Focal Loss across 3 horizons: 1h, 4h, 24h.
    L = w1 * L_1h + w4 * L_4h + w24 * L_24h
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.75,     # High alpha heavily weights positive alert examples
        weights: list = [1.0, 0.8, 0.5]
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weights = weights

    def forward(
        self,
        logits: torch.Tensor,     # [Batch, Num_Nodes, 3]
        targets: torch.Tensor     # [Batch, Num_Nodes, 3]
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Binary focal loss per horizon element
        # p_t = p if y=1 else 1-p
        p_t = targets * probs + (1 - targets) * (1 - probs)
        p_t = torch.clamp(p_t, min=1e-7, max=1.0 - 1e-7)

        # Alpha weight: alpha if y=1 else (1 - alpha)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Focal weighting factor: (1 - p_t)^gamma
        focal_weight = alpha_t * torch.pow(1.0 - p_t, self.gamma)

        # Base cross-entropy
        bce = -torch.log(p_t)
        loss_matrix = focal_weight * bce # [Batch, Num_Nodes, 3]

        # Multi-task weighted reduction
        loss_per_horizon = loss_matrix.mean(dim=(0, 1)) # [3]
        w = torch.tensor(self.weights, device=logits.device, dtype=logits.dtype)
        
        total_loss = torch.sum(loss_per_horizon * w)
        return total_loss
