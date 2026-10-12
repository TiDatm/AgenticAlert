"""
Production Real-Time Inference Runner & Live State Ring Buffer (v2).
Executes sub-15ms multi-horizon forward passes for dashboard feeds without data leakage.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
from typing import Dict, Any, List

# Ensure UTF-8 stdout handling on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root and parent are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from air_raid_forecast.src.data.build_graph import (
    REGIONS,
    REGION_TO_IDX,
    build_adjacency_matrix,
    build_directed_transition_matrices,
    get_normalized_laplacian,
    get_static_distance_features,
    compute_regional_threat_distances,
    REGIONAL_ISOCHRONES,
    REGIONAL_ISOCHRONE_STEPS,
)
from air_raid_forecast.src.data.raion_hierarchy import (
    NUM_RAIONS,
    RAION_SPEC,
    RAION_TO_IDX,
    build_cluster_assignment_matrix,
    build_raion_directed_transition_matrices,
    compute_raion_threat_distances,
    get_raions_by_oblast,
    LEAF_SPEC,
    NUM_LEAF_NODES,
    LEAF_TO_IDX,
    RAION_TO_HROMADAS,
    build_leaf_to_raion_matrix,
    compute_leaf_threat_distances,
)
from air_raid_forecast.src.data.text_features import (
    parse_batch_messages,
    parse_tactical_text,
    extract_threat_corridor,
)
from air_raid_forecast.src.models.layered_stgnn import LayeredSTGNN
from air_raid_forecast.src.models.hierarchical_stgnn import HierarchicalSTGNN
from air_raid_forecast.src.models.st_gnn import MultiResSTGNN


class AirRaidInferenceEngine:
    """
    Sub-15ms real-time prediction engine for all 25 Ukrainian regions and 136 raions.
    Supports LayeredSTGNN, HierarchicalSTGNN, and MultiResSTGNN models.
    """
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/layered_stgnn_best.pt",
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.regions = REGIONS
        self.num_nodes = len(REGIONS)

        # Adjacency matrices
        self.adj = build_adjacency_matrix()
        self.lap = get_normalized_laplacian(self.adj)
        self.lap_tensor = torch.from_numpy(self.lap).float().to(self.device)

        # Directed transition operators & static distance features
        p_fwd_np, p_bwd_np = build_directed_transition_matrices()
        self.p_fwd = p_fwd_np
        self.p_bwd = p_bwd_np
        self.p_fwd_tensor = torch.from_numpy(p_fwd_np).float().to(self.device)
        self.p_bwd_tensor = torch.from_numpy(p_bwd_np).float().to(self.device)
        self.static_dist = get_static_distance_features() # [25, 3]
        self.d_threat_reg, self.tau_hours_reg, self.tau_steps_reg = compute_regional_threat_distances()

        # Resolve checkpoint path
        resolved_ckpt = checkpoint_path
        if not os.path.exists(resolved_ckpt):
            cand = _PROJECT_ROOT / checkpoint_path
            if cand.exists():
                resolved_ckpt = str(cand)
            elif os.path.exists(os.path.join("air_raid_forecast", checkpoint_path)):
                resolved_ckpt = os.path.join("air_raid_forecast", checkpoint_path)

        # Load weights and detect model architecture
        loaded_sd = None
        if os.path.exists(resolved_ckpt):
            try:
                loaded_sd = torch.load(resolved_ckpt, map_location=self.device)
            except Exception as e:
                print(f"Notice: Failed to load state dict from {resolved_ckpt} ({e})")

        has_raion_weights = (
            loaded_sd is not None
            and any("micro" in k or "raion" in k for k in loaded_sd.keys())
        )

        if has_raion_weights or "hierarchical" in str(checkpoint_path).lower():
            self.model = HierarchicalSTGNN(
                num_macro_nodes=self.num_nodes,
                num_micro_nodes=136,
                kinetic_in_dim=16,
                campaign_in_dim=10,
                raion_in_dim=8,
                kinetic_hidden_dim=32,
                campaign_hidden_dim=32,
                raion_hidden_dim=32,
                gnn_hidden_dim=64,
                embed_dim=16
            ).to(self.device)
            self.is_layered = True
            if loaded_sd is not None:
                self.model.load_state_dict(loaded_sd, strict=False)
                print(f"Loaded trained HierarchicalSTGNN checkpoint: {resolved_ckpt}")
        else:
            self.is_layered = "layered" in str(checkpoint_path).lower() or not os.path.exists("air_raid_forecast/checkpoints/best_stgnn_model.pt")
            if self.is_layered:
                self.model = LayeredSTGNN(
                    num_nodes=self.num_nodes,
                    kinetic_in_dim=16,
                    campaign_in_dim=10,
                    kinetic_hidden_dim=32,
                    campaign_hidden_dim=32,
                    gnn_hidden_dim=64,
                    embed_dim=16
                ).to(self.device)
                if loaded_sd is not None:
                    # Filter out micro keys if loading into LayeredSTGNN
                    compat_sd = {k: v for k, v in loaded_sd.items() if k in self.model.state_dict() and v.shape == self.model.state_dict()[k].shape}
                    self.model.load_state_dict(compat_sd, strict=False)
                    print(f"Loaded trained LayeredSTGNN checkpoint: {resolved_ckpt}")
            else:
                self.model = MultiResSTGNN(num_nodes=self.num_nodes).to(self.device)
                if loaded_sd is not None:
                    self.model.load_state_dict(loaded_sd, strict=False)

        if loaded_sd is None:
            print("Running in untrained/zero-state configuration.")
        self.model.eval()

        # Micro 136-raion transition operators and geometry
        self.num_raions = NUM_RAIONS
        p_u_fwd, p_u_bwd = build_raion_directed_transition_matrices()
        self.p_micro_fwd = p_u_fwd
        self.p_micro_bwd = p_u_bwd
        self.p_micro_fwd_tensor = torch.from_numpy(p_u_fwd).float().to(self.device)
        self.p_micro_bwd_tensor = torch.from_numpy(p_u_bwd).float().to(self.device)
        self.d_threat_raion, self.tau_hours_raion, self.tau_steps_raion = compute_raion_threat_distances()
        self.raions_by_oblast = get_raions_by_oblast()
        self.s_mat = build_cluster_assignment_matrix()

        # In-memory circular buffer for 25 regions: 672 steps of 15-min intervals (7 days)
        self.history_buffer = np.zeros((672, self.num_nodes), dtype=np.float32)
        # In-memory circular buffer for 136 micro raions: 672 steps of 15-min intervals
        self.history_buffer_raion = np.zeros((672, self.num_raions), dtype=np.float32)

        # Leaf endpoints (204 nodes: 133 hromadas + 71 childless raions)
        self.num_leaf_nodes = NUM_LEAF_NODES
        self.leaf_spec = LEAF_SPEC
        self.d_threat_leaf, self.tau_hours_leaf, self.tau_steps_leaf = compute_leaf_threat_distances()
        self.leaf_to_raion_mat = build_leaf_to_raion_matrix()
        self.history_buffer_leaf = np.zeros((672, self.num_leaf_nodes), dtype=np.float32)

        self.hromada_norm_map = {}
        for l_idx, leaf in enumerate(self.leaf_spec):
            if leaf.get("type") == "hromada":
                clean = leaf["name"].strip().lower().replace("’", "'").replace("`", "'")
                self.hromada_norm_map[clean] = l_idx
                self.hromada_norm_map[leaf["name"]] = l_idx

    def update_live_state(
        self,
        current_active_regions: List[str],
        current_active_raions: Optional[List[str]] = None,
        current_active_hromadas: Optional[List[str]] = None
    ):
        """
        Updates instantaneous alert states and advances circular ring buffer for:
        1. 25 parent Oblasts
        2. 136 constituent Raions
        3. 204 fine-grained Leaf Endpoints (hromadas & childless raions)
        with Downward Hierarchical Propagation:
        Alert_leaf(t, node) = Alert_local_leaf(t, node) | Alert_parent_raion(t, parent_raion(node)) | Alert_parent_oblast(t, parent_oblast(node)).
        """
        new_step_macro = np.zeros((1, self.num_nodes), dtype=np.float32)
        new_step_micro = np.zeros((1, self.num_raions), dtype=np.float32)
        new_step_leaf = np.zeros((1, self.num_leaf_nodes), dtype=np.float32)

        # 1. Macro Oblast activations propagate downward to raions and leaf nodes
        for r_name in current_active_regions:
            if r_name in REGION_TO_IDX:
                idx = REGION_TO_IDX[r_name]
                new_step_macro[0, idx] = 1.0
                for d_idx in self.raions_by_oblast.get(r_name, []):
                    new_step_micro[0, d_idx] = 1.0
                for l_idx, leaf in enumerate(self.leaf_spec):
                    if leaf["parent_oblast_idx"] == idx:
                        new_step_leaf[0, l_idx] = 1.0

        # 2. Raion-level alerts propagate downward to child hromadas
        if current_active_raions:
            for r_name in current_active_raions:
                cleaned = r_name.strip()
                matched_d_idx = None
                if cleaned in RAION_TO_IDX:
                    matched_d_idx = RAION_TO_IDX[cleaned]
                elif (cleaned + " район") in RAION_TO_IDX:
                    matched_d_idx = RAION_TO_IDX[cleaned + " район"]
                elif cleaned.replace(" район", "").strip() in RAION_TO_IDX:
                    matched_d_idx = RAION_TO_IDX[cleaned.replace(" район", "").strip()]

                if matched_d_idx is not None:
                    new_step_micro[0, matched_d_idx] = 1.0
                    for l_idx, leaf in enumerate(self.leaf_spec):
                        if leaf["parent_raion_idx"] == matched_d_idx:
                            new_step_leaf[0, l_idx] = 1.0

        # 3. Leaf Hromada alerts activate specific leaf node (downward propagation only, no upward reflection)
        if current_active_hromadas:
            for h_name in current_active_hromadas:
                clean_h = h_name.strip().lower().replace("’", "'").replace("`", "'")
                l_idx = self.hromada_norm_map.get(clean_h)
                if l_idx is not None:
                    new_step_leaf[0, l_idx] = 1.0

        # Roll buffers left and insert newest step at the end
        self.history_buffer = np.roll(self.history_buffer, -1, axis=0)
        self.history_buffer[-1, :] = new_step_macro[0, :]

        self.history_buffer_raion = np.roll(self.history_buffer_raion, -1, axis=0)
        self.history_buffer_raion[-1, :] = new_step_micro[0, :]

        self.history_buffer_leaf = np.roll(self.history_buffer_leaf, -1, axis=0)
        self.history_buffer_leaf[-1, :] = new_step_leaf[0, :]

    def predict_realtime(
        self,
        current_hour_utc: float,
        recent_telegram_posts: List[str] = None
    ) -> Dict[str, Any]:
        """
        Executes sub-15ms model inference across all 25 regions.
        """
        # Parse tactical text tokens
        tactical_flags = parse_batch_messages(recent_telegram_posts or [])

        # 1. Kinetic Stream (32 steps = 8.0 hours high frequency)
        K = 32
        fast_slice = self.history_buffer[-K:, :] # [32, 25]
        s_front = np.zeros_like(fast_slice)      # Frontline sub-district placeholder
        s_pfwd = np.matmul(fast_slice, self.p_fwd.T) # [32, 25]
        s_pbwd = np.matmul(fast_slice, self.p_bwd.T) # [32, 25]
        s_pdiff = np.diff(s_pfwd, axis=0, prepend=s_pfwd[0:1, :])

        # Distances and flags
        dist_broad = np.tile(self.static_dist[np.newaxis, :, :], (K, 1, 1)) # [32, 25, 3]
        nat_active = np.mean(fast_slice, axis=1)
        surge_flag = (nat_active > 0.35).astype(np.float32)[:, np.newaxis] # [32, 1]
        surge_broad = np.tile(surge_flag[:, np.newaxis, :], (1, self.num_nodes, 1)) # [32, 25, 1]

        hours_kin = (np.linspace(current_hour_utc - 8.0, current_hour_utc, K)) % 24.0
        noct_flag = ((hours_kin >= 21.0) | (hours_kin <= 6.0)).astype(np.float32)[:, np.newaxis] # [32, 1]
        noct_broad = np.tile(noct_flag[:, np.newaxis, :], (1, self.num_nodes, 1)) # [32, 25, 1]

        # 6 Tactical OSINT Text Flags buffer: [K, 25, 6]
        s_text = np.zeros((K, self.num_nodes, 6), dtype=np.float32)
        if recent_telegram_posts:
            for post in recent_telegram_posts:
                flags = parse_tactical_text(post)
                corridor = extract_threat_corridor(post)
                mask = corridor["region_mask"]
                flag_vals = [
                    flags["flag_mig31k"],
                    flags["flag_ballistics"],
                    flags["flag_strat_aviation"],
                    flags["flag_shahed_group"],
                    flags["flag_kab_tactical"],
                    flags["flag_clearance"],
                ]
                for c_idx, val in enumerate(flag_vals):
                    if val > 0.0:
                        for r_idx in range(self.num_nodes):
                            if mask[r_idx] > 0.0 or c_idx == 0:  # mig31k is nationwide
                                s_text[-1, r_idx, c_idx] = max(s_text[-1, r_idx, c_idx], val)

            # Rolling decay into immediate preceding steps
            for step_back in range(1, 4):
                s_text[-1 - step_back, :, :] = s_text[-1, :, :] * (0.5 ** step_back)

        text_trans = np.transpose(s_text, (1, 0, 2)) # [25, 32, 6]

        x_kin = np.concatenate([
            np.transpose(fast_slice, (1, 0))[:, :, np.newaxis],
            np.transpose(s_front, (1, 0))[:, :, np.newaxis],
            np.transpose(s_pfwd, (1, 0))[:, :, np.newaxis],
            np.transpose(s_pbwd, (1, 0))[:, :, np.newaxis],
            np.transpose(s_pdiff, (1, 0))[:, :, np.newaxis],
            np.transpose(dist_broad, (1, 0, 2)),
            np.transpose(surge_broad, (1, 0, 2)),
            np.transpose(noct_broad, (1, 0, 2)),
            text_trans,
        ], axis=-1).astype(np.float32) # [25, 32, 16]
        x_kin_t = torch.from_numpy(x_kin).unsqueeze(0).to(self.device)

        # 2. Campaign Stream (last 24 hours: sample every 4th step backwards)
        slow_indices = [671 - 4 * k for k in range(23, -1, -1)]
        s_state = self.history_buffer[slow_indices, :]
        roll_7d_reg = np.mean(self.history_buffer, axis=0) # [25]
        roll_7d_nat = np.mean(roll_7d_reg)

        hours_camp = (np.linspace(current_hour_utc - 23.0, current_hour_utc, 24)) % 24.0
        s_sin = np.tile(np.sin(2 * np.pi * hours_camp / 24.0)[np.newaxis, :, np.newaxis], (self.num_nodes, 1, 1))
        s_cos = np.tile(np.cos(2 * np.pi * hours_camp / 24.0)[np.newaxis, :, np.newaxis], (self.num_nodes, 1, 1))
        s_h6 = np.tile(np.cos(4 * np.pi * (hours_camp - 6.0) / 24.0)[np.newaxis, :, np.newaxis], (self.num_nodes, 1, 1))
        s_nat7 = np.full((self.num_nodes, 24, 1), roll_7d_nat, dtype=np.float32)
        s_reg7 = np.tile(roll_7d_reg[:, np.newaxis, np.newaxis], (1, 24, 1))

        # Threat energy and replenishment cadence
        threat_flux = np.mean(self.history_buffer, axis=1) # [672]
        e3d_val = float(np.mean(threat_flux[-288:])) if len(threat_flux) >= 288 else float(np.mean(threat_flux))
        s_e3d = np.full((self.num_nodes, 24, 1), e3d_val, dtype=np.float32)
        s_mom = np.zeros((self.num_nodes, 24, 1), dtype=np.float32)
        s_rep = np.full((self.num_nodes, 24, 1), 0.50, dtype=np.float32)
        s_front_camp = np.zeros((self.num_nodes, 24, 1), dtype=np.float32)

        x_camp = np.concatenate([
            np.transpose(s_state, (1, 0))[:, :, np.newaxis],
            s_reg7,
            s_sin,
            s_cos,
            s_h6,
            s_nat7,
            s_e3d,
            s_mom,
            s_rep,
            s_front_camp
        ], axis=-1).astype(np.float32) # [25, 24, 10]
        x_camp_t = torch.from_numpy(x_camp).unsqueeze(0).to(self.device)

        # 3. Micro Raion Stream (32 steps = 8.0 hours across 136 raions)
        s_raion = self.history_buffer_raion[-K:, :] # [32, 136]
        s_rm_fwd = np.matmul(s_raion, self.p_micro_fwd.T)
        s_rm_bwd = np.matmul(s_raion, self.p_micro_bwd.T)
        s_rm_diff = np.diff(s_rm_fwd, axis=0, prepend=s_rm_fwd[0:1, :])

        d_broad = np.tile((self.d_threat_raion / 1000.0)[np.newaxis, :], (K, 1))
        tau_broad = np.tile((self.tau_hours_raion / 6.0)[np.newaxis, :], (K, 1))
        surge_r = np.tile(surge_flag, (1, self.num_raions))
        noct_r = np.tile(noct_flag, (1, self.num_raions))

        x_raion = np.stack([
            s_raion.T,
            s_rm_fwd.T,
            s_rm_bwd.T,
            s_rm_diff.T,
            d_broad.T,
            tau_broad.T,
            surge_r.T,
            noct_r.T,
        ], axis=-1).astype(np.float32) # [136, 32, 8]
        x_raion_t = torch.from_numpy(x_raion).unsqueeze(0).to(self.device)

        p_raion_iso = None
        tti_hours = None
        p_leaf_iso = None
        tti_leaf_hours = None
        macro_tti_hours = None

        with torch.no_grad():
            if isinstance(self.model, HierarchicalSTGNN):
                logit_1h, logit_4h, hours_24h, logit_raion_iso, pred_tti, logit_leaf_iso, pred_leaf_tti = self.model(
                    x_kin_t, x_camp_t, x_raion_t,
                    p_fwd_macro=self.p_fwd_tensor, p_bwd_macro=self.p_bwd_tensor,
                    p_fwd_micro=self.p_micro_fwd_tensor, p_bwd_micro=self.p_micro_bwd_tensor,
                    return_leaves=True
                )
                p_1h = torch.sigmoid(logit_1h).squeeze(0).cpu().numpy()
                p_4h = torch.sigmoid(logit_4h).squeeze(0).cpu().numpy()
                p_24h_hours = hours_24h.squeeze(0).cpu().numpy()
                if logit_raion_iso is not None:
                    p_raion_iso = torch.sigmoid(logit_raion_iso).squeeze(0).cpu().numpy()
                if pred_tti is not None:
                    tti_hours = pred_tti.squeeze(0).cpu().numpy()
                if logit_leaf_iso is not None:
                    p_leaf_iso = torch.sigmoid(logit_leaf_iso).squeeze(0).cpu().numpy()
                if pred_leaf_tti is not None:
                    tti_leaf_hours = pred_leaf_tti.squeeze(0).cpu().numpy()
            elif isinstance(self.model, LayeredSTGNN):
                out_layered = self.model(
                    x_kin_t, x_camp_t, self.p_fwd_tensor, self.p_bwd_tensor
                )
                if len(out_layered) == 4:
                    logit_1h, logit_4h, hours_24h, pred_macro_tti = out_layered
                    macro_tti_hours = pred_macro_tti.squeeze(0).cpu().numpy()
                else:
                    logit_1h, logit_4h, hours_24h = out_layered
                    macro_tti_hours = None
                p_1h = torch.sigmoid(logit_1h).squeeze(0).cpu().numpy()
                p_4h = torch.sigmoid(logit_4h).squeeze(0).cpu().numpy()
                p_24h_hours = hours_24h.squeeze(0).cpu().numpy()
            else:
                logits = self.model(x_kin_t[:, :, :, :2], x_camp_t[:, :, :, :3], self.lap_tensor)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                p_1h = probs[:, 0]
                p_4h = probs[:, 1]
                p_24h_hours = probs[:, 2] * 24.0

        # Tactical text boost
        if tactical_flags.get("flag_mig31k", 0) > 0:
            p_1h = np.maximum(p_1h, 0.85)

        # 5. Format Output Payload
        results = {}
        for i, reg_name in enumerate(self.regions):
            const_raions = self.raions_by_oblast.get(reg_name, [])
            reg_tau = float(round(self.tau_hours_reg[i], 2))

            # Resolve regional time-to-impact
            if self.history_buffer[-1, i] > 0.0:
                reg_tti = 0.0
                reg_prob = max(float(round(p_1h[i], 3)), 0.95)
            elif tti_hours is not None and len(const_raions) > 0:
                reg_tti = float(round(float(np.min(tti_hours[const_raions])), 2))
                reg_prob = float(round(p_1h[i], 3))
            elif macro_tti_hours is not None:
                reg_tti = float(round(float(macro_tti_hours[i]), 2))
                reg_prob = float(round(p_1h[i], 3))
            else:
                reg_tti = reg_tau
                reg_prob = float(round(p_1h[i], 3))

            # Calibrate isochrone onset probability across constituent raions if available
            if p_raion_iso is not None and len(const_raions) > 0:
                p_c = [float(p_raion_iso[d]) for d in const_raions]
                p_max = float(np.max(p_c))
                p_prod = float(np.prod([1.0 - p for p in p_c]))
                reg_iso_prob = float(round(p_max + 0.15 * ((1.0 - p_prod) - p_max), 3))
            else:
                reg_iso_prob = reg_prob

            if self.history_buffer[-1, i] > 0.0:
                reg_iso_prob = max(reg_iso_prob, 0.95)

            results[reg_name] = {
                "tactical_onset_prob_1h": reg_prob,
                "operational_corridor_prob_4h": float(round(p_4h[i], 3)),
                "expected_daily_alert_hours_24h": float(round(p_24h_hours[i], 2)),
                "time_to_impact_hours": reg_tti,
                "tau_hours": reg_tau,
                "isochrone_horizon_hours": reg_tau,
                "isochrone_onset_prob": reg_iso_prob,
                "status": "THREAT_ELEVATED" if reg_iso_prob >= 0.50 or reg_prob >= 0.50 else "NORMAL",
            }

        # 6. Fine-grained Leaf Endpoints (204 nodes: 133 hromadas + 71 childless raions)
        leaf_results = {}
        for l_idx, leaf in enumerate(self.leaf_spec):
            d_idx = leaf["parent_raion_idx"]
            o_idx = leaf["parent_oblast_idx"]
            l_prob = float(p_leaf_iso[l_idx]) if p_leaf_iso is not None else (float(p_raion_iso[d_idx]) if p_raion_iso is not None else float(p_1h[o_idx]))
            l_tti = float(tti_leaf_hours[l_idx]) if tti_leaf_hours is not None else (float(tti_hours[d_idx]) if tti_hours is not None else 6.0)
            t_tau = float(self.tau_hours_leaf[l_idx])

            # Immediate local sensor telemetry modulation
            is_active_now = float(self.history_buffer_leaf[-1, l_idx]) > 0.0
            if is_active_now:
                l_prob = max(l_prob, 0.95)
                l_tti = 0.0

            leaf_results[leaf["name"]] = {
                "id": l_idx,
                "name": leaf["name"],
                "type": leaf.get("type", "raion"),
                "node_type": leaf.get("type", "raion"),
                "parent_raion": leaf.get("parent_raion"),
                "parent_oblast": leaf.get("parent_oblast"),
                "isochrone_onset_prob": float(round(l_prob, 3)),
                "time_to_impact_hours": float(round(l_tti, 2)),
                "tau_hours": float(round(t_tau, 2)),
                "status": "THREAT_ELEVATED" if l_prob >= 0.50 else "NORMAL",
            }

        # 7. Synthesized Calibrated Raion Forecasts (136 Raions)
        raion_results = {}
        for d_idx, (r_name, r_parent, lat, lon, is_cap) in enumerate(RAION_SPEC):
            o_idx = REGION_TO_IDX.get(r_parent, 0)
            child_h = RAION_TO_HROMADAS.get(r_name, [])
            if child_h:
                c_probs = [leaf_results[h]["isochrone_onset_prob"] for h in child_h if h in leaf_results]
                c_ttis = [leaf_results[h]["time_to_impact_hours"] for h in child_h if h in leaf_results]
                if c_probs:
                    p_max = max(c_probs)
                    p_prod = float(np.prod([1.0 - p for p in c_probs]))
                    p_noisy_or = 1.0 - p_prod
                    r_prob = p_max + 0.15 * (p_noisy_or - p_max)
                    r_tti = min(c_ttis)
                else:
                    r_prob = float(p_raion_iso[d_idx]) if p_raion_iso is not None else float(p_1h[o_idx])
                    r_tti = float(tti_hours[d_idx]) if tti_hours is not None else float(self.tau_hours_raion[d_idx])
            else:
                r_leaf_dict = leaf_results.get(r_name, {})
                default_p = float(p_raion_iso[d_idx]) if p_raion_iso is not None else float(p_1h[o_idx])
                default_tti = float(tti_hours[d_idx]) if tti_hours is not None else float(self.tau_hours_raion[d_idx])
                r_prob = r_leaf_dict.get("isochrone_onset_prob", default_p)
                r_tti = r_leaf_dict.get("time_to_impact_hours", default_tti)

            # Raion immediate sensor telemetry modulation
            if float(self.history_buffer_raion[-1, d_idx]) > 0.0:
                r_prob = max(r_prob, 0.95)
                r_tti = 0.0

            raion_results[r_name] = {
                "id": d_idx,
                "parent_oblast": r_parent,
                "isochrone_onset_prob": float(round(float(r_prob), 3)),
                "time_to_impact_hours": float(round(float(r_tti), 2)),
                "tau_hours": float(round(float(self.tau_hours_raion[d_idx]), 2)),
                "is_capital": bool(is_cap),
                "child_hromadas": child_h,
                "status": "THREAT_ELEVATED" if r_prob >= 0.50 else "NORMAL",
            }

        payload: Dict[str, Any] = {
            "tactical_flags": tactical_flags,
            "region_forecasts": results,
            "raion_forecasts": raion_results,
            "leaf_forecasts": leaf_results,
            "hromada_forecasts": {
                k: v for k, v in leaf_results.items() if v.get("node_type") == "hromada"
            },
        }
        return payload


if __name__ == "__main__":
    engine = AirRaidInferenceEngine(device="cpu")
    engine.update_live_state(["Сумська область", "Харківська область"])
    out = engine.predict_realtime(
        current_hour_utc=14.5,
        recent_telegram_posts=["Зафіксовано зліт МіГ-31К!"]
    )
    print("Inference test complete.")
    print("Tactical flags parsed:", out["tactical_flags"])
    print("Kyiv City Forecast:", out["region_forecasts"]["м. Київ"])
