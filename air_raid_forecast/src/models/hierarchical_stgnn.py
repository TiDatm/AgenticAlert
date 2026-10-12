"""
Hierarchical Two-Scale Spatio-Temporal Graph Neural Network (H-STGNN).
Architected according to BLUEPRINT_HIERARCHICAL_RAION_DEFENSE_SYSTEM.md and
BLUEPRINT_KINEMATIC_ISOCHRONE_AND_SURVIVAL_DEFENSE.md.

Combines:
1. Strategic Macro Scale (25 Oblasts):
   - Campaign fatigue, national threat energy, replenishment cadence, MiG-31K alerts.
   - Directed advective macro graph diffusion with air defense interception decay.
2. Cross-Scale Hierarchy Gates:
   - Bottom-Up Tactical Pooling: D_raion^(-1) * S^T * H_micro (local frontline intensity informs campaign).
   - Top-Down Macro Projection: S * H_macro (national threat awareness conditions constituent raions).
3. Tactical Micro Scale (136 Raions):
   - High-resolution local flight-path tracking across physical borders (A_micro in R^(136 x 136)).
   - Decoupled frontline combat (Nikopol) from rear metropolis (Dnipro City).
4. Kinematic Isochrone & Survival Heads:
   - Dynamic early warning horizon tau(d) in [0.5h, 6.0h].
   - Continuous Time-to-Impact prediction T_hat(d) in [0.0, 6.0 hours].
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure UTF-8 stdout handling on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure path resolution
_CURRENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CURRENT_DIR.parents[1] if _CURRENT_DIR.parent.name == "src" else _CURRENT_DIR.parent
for _p in [str(_REPO_ROOT), str(_REPO_ROOT.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from air_raid_forecast.src.data.build_graph import (
    REGIONS,
    REGION_CENTROIDS,
    build_directed_transition_matrices,
)
from air_raid_forecast.src.data.raion_hierarchy import (
    NUM_RAIONS,
    NUM_OBLASTS,
    NUM_LEAF_NODES,
    RAION_SPEC,
    build_cluster_assignment_matrix,
    build_raion_directed_transition_matrices,
    build_leaf_to_raion_matrix,
    compute_raion_threat_distances,
    LEAF_SPEC,
)
from air_raid_forecast.src.models.layers import (
    DilatedCausalWaveNet,
    TemporalCausalGRU,
    DirectedAdvectiveGCN,
    AirDefenseInterceptionDecay,
    KinematicTrajectoryTransformer,
    ContinuousWeibullSurvivalHead,
)


class MicroRaionDirectedGCN(nn.Module):

    """
    Micro-scale graph convolutional layer for 136 raions with:
    - Directed forward threat advection (P_fwd: east/border -> west/interior)
    - Directed backward origin tracking (P_bwd)
    - Self-connection and top-down macro context injection
    """
    def __init__(self, in_features: int, out_features: int, macro_features: int = 64):
        super().__init__()
        self.w_fwd = nn.Linear(in_features, out_features, bias=False)
        self.w_bwd = nn.Linear(in_features, out_features, bias=False)
        self.w_self = nn.Linear(in_features, out_features, bias=True)
        self.w_macro = nn.Linear(macro_features, out_features, bias=False)
        self.act = nn.LeakyReLU(0.15)
        self.norm = nn.LayerNorm(out_features)

    def forward(
        self,
        h_micro: torch.Tensor,               # [B, 136, D_in]
        p_fwd: torch.Tensor,                 # [136, 136] or [B, 136, 136]
        p_bwd: torch.Tensor,                 # [136, 136] or [B, 136, 136]
        h_macro_prior: torch.Tensor          # [B, 136, D_macro]
    ) -> torch.Tensor:
        # Message passing along micro forward and backward graphs
        if p_fwd.dim() == 2:
            out_fwd = torch.matmul(p_fwd, h_micro)
            out_bwd = torch.matmul(p_bwd, h_micro)
        else:
            out_fwd = torch.bmm(p_fwd, h_micro)
            out_bwd = torch.bmm(p_bwd, h_micro)

        h_next = (
            self.w_fwd(out_fwd)
            + 0.5 * self.w_bwd(out_bwd)
            + self.w_self(h_micro)
            + self.w_macro(h_macro_prior)
        )
        return self.norm(self.act(h_next))


class HierarchicalSTGNN(nn.Module):
    def __init__(
        self,
        num_macro_nodes: int = 25,
        num_micro_nodes: int = 136,
        kinetic_in_dim: int = 16,
        campaign_in_dim: int = 10,
        raion_in_dim: int = 8,
        kinetic_hidden_dim: int = 32,
        campaign_hidden_dim: int = 32,
        raion_hidden_dim: int = 32,
        gnn_hidden_dim: int = 64,
        embed_dim: int = 16,
    ):
        super().__init__()
        self.num_macro_nodes = num_macro_nodes
        self.num_micro_nodes = num_micro_nodes

        # 1. Macro transition operators
        p_fwd_m, p_bwd_m = build_directed_transition_matrices()
        self.register_buffer("macro_p_fwd", torch.from_numpy(p_fwd_m).float())
        self.register_buffer("macro_p_bwd", torch.from_numpy(p_bwd_m).float())

        # 2. Micro transition operators & cluster assignment S
        p_fwd_u, p_bwd_u = build_raion_directed_transition_matrices()
        s_mat = build_cluster_assignment_matrix()
        self.register_buffer("micro_p_fwd", torch.from_numpy(p_fwd_u).float())
        self.register_buffer("micro_p_bwd", torch.from_numpy(p_bwd_u).float())
        self.register_buffer("cluster_s", torch.from_numpy(s_mat).float()) # [136, 25]

        # Normalized bottom-up pooling operator: D_raion^(-1) * S^T
        d_raion = s_mat.sum(axis=0, keepdims=True).T # [25, 1]
        s_pool = (s_mat.T / np.maximum(d_raion, 1.0)).astype(np.float32) # [25, 136]
        self.register_buffer("cluster_s_pool", torch.from_numpy(s_pool).float())

        # 3. Macro Encoders
        self.kinetic_encoder = DilatedCausalWaveNet(
            in_channels=kinetic_in_dim,
            hidden_channels=kinetic_hidden_dim,
            dilations=(1, 2, 4, 8, 16)
        )
        self.campaign_encoder = TemporalCausalGRU(
            input_dim=campaign_in_dim,
            hidden_dim=campaign_hidden_dim
        )
        self.bottom_up_proj = nn.Linear(raion_hidden_dim, kinetic_hidden_dim)
        self.fusion_proj = nn.Linear(kinetic_hidden_dim + campaign_hidden_dim, gnn_hidden_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
            nn.Sigmoid()
        )
        self.interception_decay = AirDefenseInterceptionDecay(in_features=gnn_hidden_dim, hidden_dim=16)

        # Macro Directed GCNs
        self.macro_gcn1 = DirectedAdvectiveGCN(
            in_features=gnn_hidden_dim,
            out_features=gnn_hidden_dim,
            num_nodes=num_macro_nodes,
            embed_dim=embed_dim
        )
        self.macro_gcn2 = DirectedAdvectiveGCN(
            in_features=gnn_hidden_dim,
            out_features=gnn_hidden_dim,
            num_nodes=num_macro_nodes,
            embed_dim=embed_dim
        )

        # Direct Pre-Launch Border Shortcut
        self.border_shortcut = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Geographic coordinates for aerodynamic bearing calculations
        coords_m = np.array([REGION_CENTROIDS[r] for r in REGIONS], dtype=np.float32)
        coords_u = np.array([[r[2], r[3]] for r in RAION_SPEC], dtype=np.float32)
        self.register_buffer("macro_coords", torch.from_numpy(coords_m).float()) # [25, 2]
        self.register_buffer("micro_coords", torch.from_numpy(coords_u).float()) # [136, 2]

        # Kinematic Trajectory Transformers (Country-Wide Receptive Field without hop limits)
        self.macro_ktt = KinematicTrajectoryTransformer(
            d_model=gnn_hidden_dim,
            num_heads=4,
            ffn_dim=128,
            dropout=0.10,
        )
        self.micro_ktt = KinematicTrajectoryTransformer(
            d_model=raion_hidden_dim,
            num_heads=4,
            ffn_dim=128,
            dropout=0.10,
        )

        # 4. Micro Raion Encoders
        self.raion_temporal_encoder = DilatedCausalWaveNet(
            in_channels=raion_in_dim,
            hidden_channels=raion_hidden_dim,
            dilations=(1, 2, 4, 8, 16)
        )
        self.micro_gcn1 = MicroRaionDirectedGCN(
            in_features=raion_hidden_dim,
            out_features=raion_hidden_dim,
            macro_features=gnn_hidden_dim
        )
        self.micro_gcn2 = MicroRaionDirectedGCN(
            in_features=raion_hidden_dim,
            out_features=raion_hidden_dim,
            macro_features=gnn_hidden_dim
        )

        # 5. Macro Output Heads
        self.head_tactical_1h = nn.Sequential(
            nn.Linear(kinetic_hidden_dim + gnn_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1)
        )
        self.head_operational_4h = nn.Sequential(
            nn.Linear(gnn_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1)
        )
        self.head_strategic_intensity = nn.Sequential(
            nn.Linear(campaign_hidden_dim + gnn_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1)
        )

        # 6. Micro Raion Heads
        # Raion Tactical Onset Head within local kinematic isochrone tau(d)
        self.head_raion_isochrone = nn.Sequential(
            nn.Linear(raion_hidden_dim + gnn_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1)
        )
        # Continuous Weibull Survival & Time-to-Impact Head [0.0, 6.0 hours]
        self.head_tti = nn.Sequential(
            nn.Linear(raion_hidden_dim + gnn_hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.weibull_tti = ContinuousWeibullSurvivalHead(
            in_features=raion_hidden_dim + gnn_hidden_dim,
            hidden_dim=48,
        )


        # 7. Discrete Leaf Heads (204 Nodes = 133 child hromadas + 71 childless raions)
        m_lr = build_leaf_to_raion_matrix()
        self.register_buffer("leaf_m_lr", torch.from_numpy(m_lr).float()) # [204, 136]
        self.num_leaf_nodes = NUM_LEAF_NODES

        # Precompute leaf indices for each raion for calibrated synthesis
        self.raion_leaf_indices = [
            [l_idx for l_idx, leaf in enumerate(LEAF_SPEC) if leaf["parent_raion_idx"] == d]
            for d in range(self.num_micro_nodes)
        ]

        self.head_leaf_isochrone = nn.Sequential(
            nn.Linear(raion_hidden_dim + gnn_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1)
        )
        self.head_leaf_tti = nn.Sequential(
            nn.Linear(raion_hidden_dim + gnn_hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def synthesize_raion_predictions(
        self,
        p_leaf_iso: torch.Tensor,
        p_leaf_tti: torch.Tensor,
        p_raion_direct: Optional[torch.Tensor] = None,
        tti_raion_direct: Optional[torch.Tensor] = None,
        lam: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Synthesizes calibrated raion predictions from discrete leaf predictions:
        - Raions with 0 mapped leaves: fallback to direct raion heads or default
        - Childless raion d with 1 leaf: P(Raion d) = P_leaf(d), TTI(Raion d) = TTI_leaf(d)
        - Raions with children C(d):
          P(Raion d) = max_{h} P_leaf(h) + lambda * ((1 - Prod(1 - P_leaf(h))) - max P_leaf(h))
          TTI(Raion d) = min_{h} TTI_leaf(h)
        """
        b = p_leaf_iso.shape[0]
        p_raion_syn = torch.zeros(b, self.num_micro_nodes, device=p_leaf_iso.device)
        tti_raion_syn = torch.zeros(b, self.num_micro_nodes, device=p_leaf_tti.device)

        for d in range(self.num_micro_nodes):
            idxs = self.raion_leaf_indices[d]
            if len(idxs) == 0:
                p_syn = p_raion_direct[:, d] if p_raion_direct is not None else torch.tensor(0.15, device=p_leaf_iso.device)
                tti_syn = tti_raion_direct[:, d] if tti_raion_direct is not None else torch.tensor(3.0, device=p_leaf_tti.device)
            elif len(idxs) == 1:
                p_l = p_leaf_iso[:, idxs[0]]
                p_syn = 0.5 * p_raion_direct[:, d] + 0.5 * p_l if p_raion_direct is not None else p_l
                tti_l = p_leaf_tti[:, idxs[0]]
                tti_syn = 0.5 * tti_raion_direct[:, d] + 0.5 * tti_l if tti_raion_direct is not None else tti_l
            else:
                p_sub = p_leaf_iso[:, idxs]
                p_max, _ = torch.max(p_sub, dim=-1)
                p_prod = torch.prod(1.0 - p_sub.clamp(0.0, 1.0 - 1e-6), dim=-1)
                p_noisy_or = 1.0 - p_prod
                p_comb = p_max + lam * (p_noisy_or - p_max)
                p_syn = 0.5 * p_raion_direct[:, d] + 0.5 * p_comb if p_raion_direct is not None else p_comb

                tti_sub = p_leaf_tti[:, idxs]
                tti_min, _ = torch.min(tti_sub, dim=-1)
                tti_syn = 0.5 * tti_raion_direct[:, d] + 0.5 * tti_min if tti_raion_direct is not None else tti_min

            p_raion_syn[:, d] = p_syn.clamp(0.0, 1.0)
            tti_raion_syn[:, d] = tti_syn

        return p_raion_syn, tti_raion_syn

    def forward(
        self,
        x_kin: torch.Tensor,                                # [B, 25, 32, 16]
        x_camp: torch.Tensor,                               # [B, 25, 24, 10]
        x_raion: Optional[torch.Tensor] = None,             # [B, 136, 32, 8]
        p_fwd_macro: Optional[torch.Tensor] = None,
        p_bwd_macro: Optional[torch.Tensor] = None,
        p_fwd_micro: Optional[torch.Tensor] = None,
        p_bwd_micro: Optional[torch.Tensor] = None,
        x_leaf: Optional[torch.Tensor] = None,              # [B, 204, 32, 8]
        return_leaves: bool = False,
    ):
        """
        Forward pass.
        Returns:
        - logit_1h: [B, 25]
        - logit_4h: [B, 25]
        - hours_24h: [B, 25]
        - logit_raion_iso: [B, 136] (if x_raion is provided)
        - pred_tti: [B, 136] (continuous hours in [0.0, 6.0], if x_raion is provided)
        """
        b = x_kin.shape[0]
        if p_fwd_macro is None:
            p_fwd_macro = self.macro_p_fwd
        if p_bwd_macro is None:
            p_bwd_macro = self.macro_p_bwd
        if p_fwd_micro is None:
            p_fwd_micro = self.micro_p_fwd
        if p_bwd_micro is None:
            p_bwd_micro = self.micro_p_bwd

        # 1. Micro Temporal Processing & Bottom-Up Pooling (if available)
        h_raion = None
        bottom_up_inj = 0.0
        if x_raion is not None:
            h_raion = self.raion_temporal_encoder(x_raion) # [B, 136, 32]
            # Bottom-Up Tactical Pooling: D_raion^(-1) * S^T * H_micro -> [B, 25, 32]
            pooled_micro = torch.matmul(self.cluster_s_pool, h_raion) # [B, 25, 32]
            bottom_up_inj = self.bottom_up_proj(pooled_micro)

        # 2. Macro Temporal Encodings
        h_kin = self.kinetic_encoder(x_kin) # [B, 25, 32]
        if isinstance(bottom_up_inj, torch.Tensor):
            h_kin = h_kin + 0.3 * bottom_up_inj # Informs macro state of frontline ground combat

        h_camp = self.campaign_encoder(x_camp) # [B, 25, 32]

        # 3. Macro Multi-Scale Fusion & Directed GCN
        h_cat = torch.cat([h_kin, h_camp], dim=-1) # [B, 25, 64]
        h_fused = self.fusion_proj(h_cat)
        gate = self.fusion_gate(h_fused)
        h_fused = gate * h_fused

        p_fwd_eff = self.interception_decay(h_fused, p_fwd_macro)
        h_spat_macro = self.macro_gcn1(h_fused, p_fwd_eff, p_bwd_macro)
        h_spat_macro = self.macro_gcn2(h_spat_macro, p_fwd_eff, p_bwd_macro) # [B, 25, 64]
        # Country-Wide Kinematic Trajectory Transformer at Macro Scale:
        h_spat_macro = self.macro_ktt(h_spat_macro, coords=self.macro_coords)

        # 4. Macro Decoupled Projections
        h_tactical = torch.cat([h_kin, h_spat_macro], dim=-1)
        logit_1h = self.head_tactical_1h(h_tactical).squeeze(-1) # [B, 25]

        # Instantaneous Border Shortcut
        if x_kin.shape[-1] >= 16:
            tactical_sig = x_kin[:, :, -1, [10, 11, 14]]
            border_boost = self.border_shortcut(tactical_sig).squeeze(-1)
            logit_1h = logit_1h + border_boost

        logit_4h = self.head_operational_4h(h_spat_macro).squeeze(-1) # [B, 25]

        h_strat = torch.cat([h_camp, h_spat_macro], dim=-1)
        raw_hours = self.head_strategic_intensity(h_strat).squeeze(-1)
        hours_24h = torch.sigmoid(raw_hours) * 24.0 # [B, 25]

        # 5. Micro Tactical Processing & Discrete Leaf Projections
        logit_raion_iso = None
        pred_tti = None
        logit_leaf_iso = None
        pred_leaf_tti = None

        if h_raion is not None:
            # Top-Down Projection: S * H_macro -> [B, 136, 64]
            h_prior_raion = torch.matmul(self.cluster_s, h_spat_macro)

            # Micro Graph Diffusion across 136 raions
            h_micro = self.micro_gcn1(h_raion, p_fwd_micro, p_bwd_micro, h_prior_raion)
            h_micro = self.micro_gcn2(h_micro, p_fwd_micro, p_bwd_micro, h_prior_raion) # [B, 136, 32]
            # Country-Wide Kinematic Trajectory Transformer at Micro Scale:
            h_micro = self.micro_ktt(h_micro, coords=self.micro_coords)

            # Fused raion state
            h_raion_fused = torch.cat([h_micro, h_prior_raion], dim=-1) # [B, 136, 96]

            # Direct neural prediction heads for all 136 raions
            logit_raion_direct = self.head_raion_isochrone(h_raion_fused).squeeze(-1) # [B, 136]
            raw_tti_legacy = self.head_tti(h_raion_fused).squeeze(-1) # [B, 136]
            pred_tti_weibull, _, _ = self.weibull_tti(h_raion_fused) # [B, 136]
            pred_tti_direct = 0.5 * (torch.sigmoid(raw_tti_legacy) * 6.0) + 0.5 * pred_tti_weibull # [B, 136]
            p_raion_direct = torch.sigmoid(logit_raion_direct)


            # Leaf state projection via M_lr in R^(204 x 136)
            h_leaf_fused = torch.matmul(self.leaf_m_lr, h_raion_fused) # [B, 204, 96]
            logit_leaf_iso = self.head_leaf_isochrone(h_leaf_fused).squeeze(-1) # [B, 204]
            p_leaf_iso = torch.sigmoid(logit_leaf_iso)
            raw_leaf_tti = self.head_leaf_tti(h_leaf_fused).squeeze(-1) # [B, 204]
            pred_leaf_tti = torch.sigmoid(raw_leaf_tti) * 6.0 # [B, 204]

            # Calibrated Multi-Instance Blending for all 136 raions
            p_raion_syn, tti_raion_syn = self.synthesize_raion_predictions(
                p_leaf_iso, pred_leaf_tti, p_raion_direct=p_raion_direct, tti_raion_direct=pred_tti_direct, lam=0.15
            )
            logit_raion_iso = torch.logit(p_raion_syn.clamp(1e-6, 1.0 - 1e-6)) # [B, 136]
            pred_tti = tti_raion_syn # [B, 136]

        if return_leaves:
            return logit_1h, logit_4h, hours_24h, logit_raion_iso, pred_tti, logit_leaf_iso, pred_leaf_tti

        return logit_1h, logit_4h, hours_24h, logit_raion_iso, pred_tti


class HierarchicalDefenseLoss(nn.Module):
    """
    Unified Hierarchical Multi-Scale Loss with Prevalence-Balanced Reductions:
    1. Masked Macro 1h Focal Loss (idle oblasts)
    2. Masked Macro 4h Focal Loss (idle oblasts)
    3. Normalized Smooth L1 Loss on Strategic Daily Hours
    4. Per-Raion Masked Isochrone Focal Loss (prevents rear gradient starvation)
    5. Continuous Survival / Time-to-Impact Loss on Micro Raions
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.70):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.huber = nn.SmoothL1Loss(reduction="mean")

    def forward(
        self,
        pred_1h: torch.Tensor,                # [B, 25]
        pred_4h: torch.Tensor,                # [B, 25]
        pred_hours: torch.Tensor,             # [B, 25]
        target_1h: torch.Tensor,              # [B, 25]
        target_4h: torch.Tensor,              # [B, 25]
        target_hours: torch.Tensor,           # [B, 25]
        idle_mask_macro: torch.Tensor,        # [B, 25]
        pred_raion_iso: Optional[torch.Tensor] = None,    # [B, 136]
        pred_tti: Optional[torch.Tensor] = None,          # [B, 136]
        target_raion_iso: Optional[torch.Tensor] = None,  # [B, 136]
        target_tti: Optional[torch.Tensor] = None,        # [B, 136]
        idle_mask_raion: Optional[torch.Tensor] = None,   # [B, 136]
        pred_leaf_iso: Optional[torch.Tensor] = None,     # [B, 204]
        pred_leaf_tti: Optional[torch.Tensor] = None,     # [B, 204]
        target_leaf_iso: Optional[torch.Tensor] = None,   # [B, 204]
        target_leaf_tti: Optional[torch.Tensor] = None,   # [B, 204]
        idle_mask_leaf: Optional[torch.Tensor] = None,    # [B, 204]
        return_components: bool = False,
    ) -> torch.Tensor:
        # 1. Macro 1h
        prob_1h = torch.sigmoid(pred_1h).clamp(1e-7, 1.0 - 1e-7)
        pt_1 = target_1h * prob_1h + (1.0 - target_1h) * (1.0 - prob_1h)
        alpha_t1 = target_1h * self.alpha + (1.0 - target_1h) * (1.0 - self.alpha)
        focal_1 = alpha_t1 * torch.pow(1.0 - pt_1, self.gamma) * (-torch.log(pt_1))
        loss_macro_1 = (focal_1 * idle_mask_macro).sum() / idle_mask_macro.sum().clamp(min=1.0)

        # 2. Macro 4h
        prob_4h = torch.sigmoid(pred_4h).clamp(1e-7, 1.0 - 1e-7)
        pt_4 = target_4h * prob_4h + (1.0 - target_4h) * (1.0 - prob_4h)
        alpha_t4 = target_4h * self.alpha + (1.0 - target_4h) * (1.0 - self.alpha)
        focal_4 = alpha_t4 * torch.pow(1.0 - pt_4, self.gamma) * (-torch.log(pt_4))
        loss_macro_4 = (focal_4 * idle_mask_macro).sum() / idle_mask_macro.sum().clamp(min=1.0)

        # 3. Macro 24h
        loss_macro_hours = self.huber(pred_hours / 24.0, target_hours / 24.0)

        loss_tactical = 1.0 * loss_macro_1 + 0.8 * loss_macro_4
        loss_continuous = 1.0 * loss_macro_hours

        # 4. Micro Raion Losses with Per-Node Balance
        if pred_raion_iso is not None and target_raion_iso is not None and idle_mask_raion is not None:
            prob_r = torch.sigmoid(pred_raion_iso).clamp(1e-7, 1.0 - 1e-7)
            pt_r = target_raion_iso * prob_r + (1.0 - target_raion_iso) * (1.0 - prob_r)
            alpha_tr = target_raion_iso * self.alpha + (1.0 - target_raion_iso) * (1.0 - self.alpha)
            focal_r = alpha_tr * torch.pow(1.0 - pt_r, self.gamma) * (-torch.log(pt_r))

            # Per-raion masked mean avoids rear gradient starvation
            idle_per_raion = idle_mask_raion.sum(dim=0).clamp(min=1.0) # [136]
            loss_per_raion = (focal_r * idle_mask_raion).sum(dim=0) / idle_per_raion # [136]
            loss_raion_iso = loss_per_raion.mean()
            loss_tactical = loss_tactical + 0.8 * loss_raion_iso

            # Continuous Time-to-Impact Loss
            if pred_tti is not None and target_tti is not None:
                # Event mask: idle raion experiencing onset within 6h (target_tti < 6.0)
                event_mask = idle_mask_raion * (target_tti < 6.0).float()
                # Censored mask: idle raion with no alert in 6h (target_tti >= 6.0)
                censor_mask = idle_mask_raion * (target_tti >= 6.0).float()

                # Huber loss on observed events (scaled by 6h)
                l_event = F.smooth_l1_loss(pred_tti / 6.0, target_tti / 6.0, reduction="none")
                loss_event = (l_event * event_mask).sum() / event_mask.sum().clamp(min=1.0)

                # Right-censored penalty: penalize predicting early if actual >= 6.0h
                l_censor = F.relu(5.5 - pred_tti) / 6.0
                loss_censor = (l_censor * censor_mask).sum() / censor_mask.sum().clamp(min=1.0)

                loss_tti = loss_event + 0.3 * loss_censor
                loss_continuous = loss_continuous + 0.5 * loss_tti

        # 5. Discrete Leaf Losses (204 Leaves)
        if pred_leaf_iso is not None and target_leaf_iso is not None and idle_mask_leaf is not None:
            prob_l = torch.sigmoid(pred_leaf_iso).clamp(1e-7, 1.0 - 1e-7)
            pt_l = target_leaf_iso * prob_l + (1.0 - target_leaf_iso) * (1.0 - prob_l)
            alpha_tl = target_leaf_iso * self.alpha + (1.0 - target_leaf_iso) * (1.0 - self.alpha)
            focal_l = alpha_tl * torch.pow(1.0 - pt_l, self.gamma) * (-torch.log(pt_l))

            idle_per_leaf = idle_mask_leaf.sum(dim=0).clamp(min=1.0)
            loss_per_leaf = (focal_l * idle_mask_leaf).sum(dim=0) / idle_per_leaf
            loss_leaf_iso = loss_per_leaf.mean()
            loss_tactical = loss_tactical + 0.8 * loss_leaf_iso

            if pred_leaf_tti is not None and target_leaf_tti is not None:
                event_mask_l = idle_mask_leaf * (target_leaf_tti < 6.0).float()
                censor_mask_l = idle_mask_leaf * (target_leaf_tti >= 6.0).float()

                l_event_l = F.smooth_l1_loss(pred_leaf_tti / 6.0, target_leaf_tti / 6.0, reduction="none")
                loss_event_l = (l_event_l * event_mask_l).sum() / event_mask_l.sum().clamp(min=1.0)

                l_censor_l = F.relu(5.5 - pred_leaf_tti) / 6.0
                loss_censor_l = (l_censor_l * censor_mask_l).sum() / censor_mask_l.sum().clamp(min=1.0)

                loss_leaf_tti = loss_event_l + 0.3 * loss_censor_l
                loss_continuous = loss_continuous + 0.5 * loss_leaf_tti

        loss_total = loss_tactical + loss_continuous
        if return_components:
            return loss_total, loss_tactical, loss_continuous
        return loss_total
