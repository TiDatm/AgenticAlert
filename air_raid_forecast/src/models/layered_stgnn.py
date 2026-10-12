"""
Layered Spatio-Temporal Graph Neural Network (Layered ST-GNN v2).
Decouples defense modeling into three specialized output representations:
1. Tactical Head (1h): Binary cross-entropy on genuine idle-to-alert transitions with local kinetic bypass
2. Operational Head (4h): Penetration probability across directed threat corridors (P_fwd, P_bwd)
3. Strategic Head (24h): Normalized continuous daily alert hours [0 - 24h] with campaign momentum
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from air_raid_forecast.src.data.build_graph import build_directed_transition_matrices
from air_raid_forecast.src.models.layers import (
    DilatedCausalWaveNet,
    TemporalCausalGRU,
    DirectedAdvectiveGCN,
    AirDefenseInterceptionDecay,
)


class LayeredSTGNN(nn.Module):
    def __init__(
        self,
        num_nodes: int = 25,
        kinetic_in_dim: int = 16,   # [state, frontline, pfwd, pbwd, pdiff, d_rf, d_fl, d_bs, surge, nocturnal, mig31k, ballistics, strat_aviation, shahed, kab, clearance]
        campaign_in_dim: int = 10,  # [state, roll_7d, sin_h, cos_h, h6, nat_7d, e3d, mom, rep, front]
        kinetic_hidden_dim: int = 32,
        campaign_hidden_dim: int = 32,
        gnn_hidden_dim: int = 64,
        embed_dim: int = 16,
    ):
        super().__init__()
        self.num_nodes = num_nodes

        # Default directed transition matrices (P_fwd, P_bwd)
        p_fwd_np, p_bwd_np = build_directed_transition_matrices()
        self.register_buffer("default_p_fwd", torch.from_numpy(p_fwd_np).float())
        self.register_buffer("default_p_bwd", torch.from_numpy(p_bwd_np).float())

        # 1. High-Frequency Kinetic Encoder: 5-layer Dilated Causal WaveNet (RF = 32 steps = 8.0 hours)
        self.kinetic_encoder = DilatedCausalWaveNet(
            in_channels=kinetic_in_dim,
            hidden_channels=kinetic_hidden_dim,
            dilations=(1, 2, 4, 8, 16)
        )

        # 2. Macro-Frequency Campaign Encoder: Unidirectional Causal GRU (24 hours)
        self.campaign_encoder = TemporalCausalGRU(
            input_dim=campaign_in_dim,
            hidden_dim=campaign_hidden_dim
        )

        # 3. Multi-Scale Fusion Gate
        self.fusion_proj = nn.Linear(kinetic_hidden_dim + campaign_hidden_dim, gnn_hidden_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
            nn.Sigmoid()
        )

        # Physical Air Defense Interception Decay (models shoot-down attrition along flight corridors)
        self.interception_decay = AirDefenseInterceptionDecay(in_features=gnn_hidden_dim, hidden_dim=16)

        # 4. Directed Advective Graph Message Passing (dual P_fwd / P_bwd + adaptive corridor discovery)
        self.gcn1 = DirectedAdvectiveGCN(
            in_features=gnn_hidden_dim,
            out_features=gnn_hidden_dim,
            num_nodes=num_nodes,
            embed_dim=embed_dim
        )
        self.gcn2 = DirectedAdvectiveGCN(
            in_features=gnn_hidden_dim,
            out_features=gnn_hidden_dim,
            num_nodes=num_nodes,
            embed_dim=embed_dim
        )

        # Direct Pre-Launch Border Shortcut:
        # Bypasses multi-hop graph diffusion for instantaneous ballistic / MiG-31K / KAB strikes
        self.border_shortcut = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # 5. Specialized Decoupled Output Heads
        # Head 1: Tactical Onset (1h) & Continuous Time-to-Impact [0.0 - 6.0h]
        self.head_tactical = nn.Sequential(
            nn.Linear(kinetic_hidden_dim + gnn_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1)
        )
        self.head_tti = nn.Sequential(
            nn.Linear(kinetic_hidden_dim + gnn_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1)
        )

        # Head 2: Operational Corridor Wave (4h) - macro flight-path propagation
        self.head_operational = nn.Sequential(
            nn.Linear(gnn_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1)
        )

        # Head 3: Strategic Campaign Intensity (24h) - outputs continuous daily hours [0.0 - 24.0]
        # Uses campaign temporal context + graph context with sigmoid normalization
        self.head_strategic_intensity = nn.Sequential(
            nn.Linear(campaign_hidden_dim + gnn_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1)
        )

    def forward(
        self,
        x_kin: torch.Tensor,                                # [B, 25, 32, 16]
        x_camp: torch.Tensor,                               # [B, 25, 24, 10]
        p_fwd: Optional[torch.Tensor] = None,               # [25, 25] or [B, 25, 25]
        p_bwd: Optional[torch.Tensor] = None,               # [25, 25] or [B, 25, 25]
        static_lap: Optional[torch.Tensor] = None,          # Backward compatibility
        return_tti: bool = True
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        Returns:
        - logit_tactical_1h: [B, 25]
        - logit_operational_4h: [B, 25]
        - pred_intensity_24h: [B, 25] (continuous hours in [0, 24])
        - pred_tti: [B, 25] (continuous time-to-impact hours in [0.0, 6.0], if return_tti=True)
        """
        # Resolve transition matrices
        if p_fwd is None:
            p_fwd = self.default_p_fwd
        if p_bwd is None:
            p_bwd = self.default_p_bwd

        # 1. Temporal encodings
        h_kin = self.kinetic_encoder(x_kin)     # [B, 25, 32]
        h_camp = self.campaign_encoder(x_camp)  # [B, 25, 32]

        # 2. Multi-scale fusion
        h_cat = torch.cat([h_kin, h_camp], dim=-1) # [B, 25, 64]
        h_fused = self.fusion_proj(h_cat)
        gate = self.fusion_gate(h_fused)
        h_fused = gate * h_fused

        # 3. Air Defense Interception Attrition along Flight Corridors
        p_fwd_eff = self.interception_decay(h_fused, p_fwd)

        # Directed Advective Graph Message Passing with Attrition
        h_spat = self.gcn1(h_fused, p_fwd_eff, p_bwd)
        h_spat = self.gcn2(h_spat, p_fwd_eff, p_bwd)

        # 4. Decoupled output projections
        # Tactical head uses local kinetic bypass to prevent spatial over-smoothing
        h_tactical = torch.cat([h_kin, h_spat], dim=-1) # [B, 25, 96]
        logit_1h = self.head_tactical(h_tactical).squeeze(-1) # [B, 25]

        # Direct Pre-Launch Border Shortcut for Ballistics / MiG-31K / KAB
        if x_kin.shape[-1] >= 16:
            tactical_sig = x_kin[:, :, -1, [10, 11, 14]] # [B, 25, 3]
            border_boost = self.border_shortcut(tactical_sig).squeeze(-1) # [B, 25]
            logit_1h = logit_1h + border_boost

        # Continuous Time-to-Impact prediction in [0.0, 6.0 hours]
        raw_tti = self.head_tti(h_tactical).squeeze(-1) # [B, 25]
        pred_tti = torch.sigmoid(raw_tti) * 6.0 # Range: [0.0, 6.0]

        # Operational head uses corridor advection
        logit_4h = self.head_operational(h_spat).squeeze(-1) # [B, 25]

        # Strategic head uses campaign momentum + graph context with sigmoid normalization
        h_strat = torch.cat([h_camp, h_spat], dim=-1) # [B, 25, 96]
        raw_hours = self.head_strategic_intensity(h_strat).squeeze(-1) # [B, 25]
        hours_24h = torch.sigmoid(raw_hours) * 24.0 # Range: [0.0, 24.0]

        if return_tti:
            return logit_1h, logit_4h, hours_24h, pred_tti
        return logit_1h, logit_4h, hours_24h


class LayeredDefenseLoss(nn.Module):
    """
    Combines:
    1. Masked Focal Loss on Tactical Onset (computed ONLY on currently idle regions)
    2. Masked Focal Loss on Operational Incursion (computed ONLY on currently idle regions)
    3. Normalized Smooth L1 Loss on Strategic Daily Intensity (scaled to [0, 1] to prevent gradient domination)
    4. Continuous Survival / Time-to-Onset Smooth L1 Loss on Observed Incursions with Right-Censored Penalty
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.70):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.huber = nn.SmoothL1Loss(reduction="mean")

    def forward(
        self,
        pred_1h: torch.Tensor,                    # Logit [B, 25]
        pred_4h: torch.Tensor,                    # Logit [B, 25]
        pred_hours: torch.Tensor,                 # Continuous [B, 25] in [0, 24]
        target_1h: torch.Tensor,                  # Binary [B, 25]
        target_4h: torch.Tensor,                  # Binary [B, 25]
        target_hours: torch.Tensor,               # Continuous [B, 25] in [0, 24]
        idle_mask: torch.Tensor,                  # Binary mask [B, 25]
        pred_tti: Optional[torch.Tensor] = None,   # Continuous [B, 25] in [0, 6]
        target_tti: Optional[torch.Tensor] = None # Continuous [B, 25] in [0, 6]
    ) -> torch.Tensor:
        # 1. Masked Tactical Onset Loss
        prob_1h = torch.sigmoid(pred_1h)
        prob_1h = torch.clamp(prob_1h, 1e-7, 1.0 - 1e-7)
        pt_1 = target_1h * prob_1h + (1.0 - target_1h) * (1.0 - prob_1h)
        alpha_t1 = target_1h * self.alpha + (1.0 - target_1h) * (1.0 - self.alpha)
        focal_1 = alpha_t1 * torch.pow(1.0 - pt_1, self.gamma) * (-torch.log(pt_1))
        # Mask only idle regions (eliminates frontline continuous state trap!)
        masked_focal_1 = (focal_1 * idle_mask).sum() / torch.clamp(idle_mask.sum(), min=1.0)

        # 2. Masked Operational Corridor Loss
        prob_4h = torch.sigmoid(pred_4h)
        prob_4h = torch.clamp(prob_4h, 1e-7, 1.0 - 1e-7)
        pt_4 = target_4h * prob_4h + (1.0 - target_4h) * (1.0 - prob_4h)
        alpha_t4 = target_4h * self.alpha + (1.0 - target_4h) * (1.0 - self.alpha)
        focal_4 = alpha_t4 * torch.pow(1.0 - pt_4, self.gamma) * (-torch.log(pt_4))
        masked_focal_4 = (focal_4 * idle_mask).sum() / torch.clamp(idle_mask.sum(), min=1.0)

        # 3. Strategic 24h Alert Intensity Loss (Normalized to [0, 1] to eliminate the 0.23 Bayes floor)
        norm_pred_hours = pred_hours / 24.0
        norm_target_hours = target_hours / 24.0
        loss_intensity = self.huber(norm_pred_hours, norm_target_hours)

        # 4. Continuous Time-to-Onset Hazard / Survival Loss
        loss_tti = torch.tensor(0.0, device=pred_1h.device)
        if pred_tti is not None and target_tti is not None:
            # Event mask: idle regions experiencing onset within 6h (target_tti < 6.0)
            event_mask = idle_mask * (target_tti < 6.0).float()
            # Censored mask: idle regions with no alert in 6h (target_tti >= 6.0)
            censor_mask = idle_mask * (target_tti >= 6.0).float()

            # Huber / Smooth L1 loss on observed events (scaled by 6h)
            l_event = F.smooth_l1_loss(pred_tti / 6.0, target_tti / 6.0, reduction="none")
            loss_event = (l_event * event_mask).sum() / event_mask.sum().clamp(min=1.0)

            # Right-censored penalty: penalize predicting early if actual >= 6.0h
            l_censor = F.relu(5.5 - pred_tti) / 6.0
            loss_censor = (l_censor * censor_mask).sum() / censor_mask.sum().clamp(min=1.0)

            loss_tti = loss_event + 0.3 * loss_censor

        # Balanced multi-tier loss: all components in comparable magnitude (~0.02 - 0.10)
        total_loss = 1.0 * masked_focal_1 + 0.8 * masked_focal_4 + 1.0 * loss_intensity + 0.5 * loss_tti
        return total_loss


# Unified Canonical Architecture: Hierarchical De-biased ST-GNN
from air_raid_forecast.src.models.hierarchical_stgnn import (
    HierarchicalSTGNN,
    HierarchicalDefenseLoss,
)
HierarchicalLayeredSTGNN = HierarchicalSTGNN

