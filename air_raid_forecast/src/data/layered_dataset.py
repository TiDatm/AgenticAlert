"""
Three-Tier Layered Dataset Generator with Strict Anti-Leakage Controls:
Layer 1: Tactical Onset (1h, 4 steps) -> P(New Siren Starts | Currently Idle)
Layer 2: Operational Corridor Penetration (4h, 16 steps) -> P(New Wave Incurs | Currently Idle)
Layer 3: Strategic Campaign Intensity (24h, 96 steps) -> Expected Alert Hours [0.0 - 24.0]
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

current_file_path = Path(__file__).resolve()
repo_root = current_file_path.parents[2] if "air_raid_forecast" in [p.name for p in current_file_path.parents] else current_file_path.parent
for p in [repo_root, repo_root.parent]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from air_raid_forecast.src.data.build_graph import (
    REGIONS, REGION_TO_IDX, build_adjacency_matrix,
    build_directed_transition_matrices, get_static_distance_features,
    compute_regional_threat_distances, REGIONAL_ISOCHRONES, REGIONAL_ISOCHRONE_STEPS,
)
from air_raid_forecast.src.data.raion_hierarchy import (
    NUM_RAIONS,
    NUM_LEAF_NODES,
    build_raion_directed_transition_matrices,
    compute_raion_threat_distances,
    build_leaf_to_raion_matrix,
    compute_leaf_threat_distances,
)


class LayeredDefenseDataset:
    """
    Produces strictly causal multi-tier tensors:
    - X_kinetic: [Batch, 25, 32, 16] (15m high-frequency tactical steps over past 8.0 hours fused with 6 OSINT flags)
    - X_campaign: [Batch, 25, 24, 10] (1h macro-frequency steps over past 24.0 hours)
    - X_raion: [Batch, 136, 32, 8] (15m high-frequency steps over past 8.0 hours across 136 raions)
    - Y_tactical_1h: [Batch, 25] (Binary onset in next 1h)
    - Y_operational_4h: [Batch, 25] (Binary onset in next 4h)
    - Y_strategic_24h: [Batch, 25] (Continuous hours of alert in next 24h, [0.0 - 24.0])
    - Idle_mask: [Batch, 25] (1 if region is currently idle at time t, 0 if already on alert)
    - Y_raion_iso: [Batch, 136] (Binary onset in individual raion isochrone horizon tau(d))
    - Y_tti: [Batch, 136] (Continuous hours to arrival in [0.0, 6.0])
    - Idle_mask_raion: [Batch, 136] (1 if raion is currently idle at time t)
    """
    def __init__(
        self,
        grid_15m: np.ndarray,
        time_idx: pd.DatetimeIndex,
        adj_matrix: np.ndarray,
        grid_front: Optional[np.ndarray] = None,
        grid_text: Optional[np.ndarray] = None,
        grid_raion: Optional[np.ndarray] = None,
        grid_leaf: Optional[np.ndarray] = None,
        sample_stride: int = 4,
        stratified: bool = False
    ):
        self.grid = grid_15m
        self.time_idx = time_idx
        self.adj = adj_matrix
        self.sample_stride = sample_stride
        self.n_steps, self.n_regions = grid_15m.shape

        # Frontline context grid [N, 25]
        if grid_front is not None and grid_front.shape == grid_15m.shape:
            self.grid_front = grid_front
        else:
            self.grid_front = np.zeros_like(grid_15m)

        # Tactical text OSINT indicators [N, 25, 6]
        if grid_text is not None and grid_text.shape == (self.n_steps, self.n_regions, 6):
            self.grid_text = grid_text.astype(np.float32)
        else:
            from air_raid_forecast.src.data.telegram_dataset import build_tactical_text_grid
            self.grid_text = build_tactical_text_grid(time_idx, grid_strat=self.grid, grid_front=self.grid_front)

        # Micro 136-Raion Grid Setup
        self.has_raions = False
        if grid_raion is not None and grid_raion.shape[1] == NUM_RAIONS:
            self.grid_raion = grid_raion.astype(np.float32)
            self.has_raions = True
            p_u_fwd, p_u_bwd = build_raion_directed_transition_matrices()
            self.p_micro_fwd = p_u_fwd
            self.p_micro_bwd = p_u_bwd
            self.p_micro_fwd_active = np.matmul(self.grid_raion, self.p_micro_fwd.T)
            self.p_micro_bwd_active = np.matmul(self.grid_raion, self.p_micro_bwd.T)
            self.p_micro_fwd_diff = np.diff(self.p_micro_fwd_active, axis=0, prepend=self.p_micro_fwd_active[0:1, :])
            self.d_threat_raion, self.tau_hours_raion, self.tau_steps_raion = compute_raion_threat_distances()
            self.onsets_raion = (np.diff(self.grid_raion, axis=0, prepend=self.grid_raion[0:1, :]) > 0).astype(np.float32)

        # Discrete 204-Leaf Grid Setup (133 active child hromadas + 71 childless raions)
        self.has_leaves = False
        self.m_leaf_raion = build_leaf_to_raion_matrix() # [204, 136]
        if grid_leaf is not None and grid_leaf.shape[1] == NUM_LEAF_NODES:
            self.grid_leaf = grid_leaf.astype(np.float32)
            self.has_leaves = True
        elif self.has_raions:
            self.grid_leaf = np.matmul(self.grid_raion, self.m_leaf_raion.T).astype(np.float32)
            self.has_leaves = True

        if self.has_leaves:
            self.d_threat_leaf, self.tau_hours_leaf, self.tau_steps_leaf = compute_leaf_threat_distances()
            self.onsets_leaf = (np.diff(self.grid_leaf, axis=0, prepend=self.grid_leaf[0:1, :]) > 0).astype(np.float32)

        # 1. Directed transition operators (P_fwd, P_bwd)
        p_fwd, p_bwd = build_directed_transition_matrices()
        self.p_fwd = p_fwd
        self.p_bwd = p_bwd

        # 2. Static normalized threat distances [25, 3]
        # [d_border, d_frontline, d_coast]
        self.static_dist = get_static_distance_features()
        self.d_threat_reg, self.tau_hours_reg, self.tau_steps_reg = compute_regional_threat_distances()

        # 3. Directed dynamic threat pressure
        # P_fwd * X: Upstream incursion pressure
        self.p_fwd_active = np.matmul(self.grid, self.p_fwd.T)     # [N, 25]
        self.p_bwd_active = np.matmul(self.grid, self.p_bwd.T)     # [N, 25]
        self.p_fwd_diff = np.diff(self.p_fwd_active, axis=0, prepend=self.p_fwd_active[0:1, :])

        # 4. Strategic Onsets (clean transitions 0 -> 1 on strategic sirens)
        diffs = np.diff(self.grid, axis=0, prepend=self.grid[0:1, :])
        self.onsets = (diffs > 0).astype(np.float32)

        # 5. Temporal harmonics
        hours = np.array([t.hour + t.minute / 60.0 for t in time_idx])
        self.sin_h = np.sin(2 * np.pi * hours / 24.0).astype(np.float32)
        self.cos_h = np.cos(2 * np.pi * hours / 24.0).astype(np.float32)
        self.harmonic_6h = np.cos(4 * np.pi * (hours - 6.0) / 24.0).astype(np.float32)

        # Tactical surge & nocturnal flags
        nat_active = np.mean(self.grid, axis=1) # [N]
        self.surge_flag = (nat_active > 0.35).astype(np.float32)[:, np.newaxis] # [N, 1]
        self.nocturnal_flag = ((hours >= 21.0) | (hours <= 6.0)).astype(np.float32)[:, np.newaxis] # [N, 1]

        # 6. Rolling 7-day fatigue
        df_grid = pd.DataFrame(self.grid)
        self.roll_7d_reg = df_grid.rolling(window=672, min_periods=1).mean().values.astype(np.float32)
        self.roll_7d_nat = np.mean(self.roll_7d_reg, axis=1, keepdims=True) # [N, 1]

        # 7. Threat Energy & Logistics Replenishment Cadence
        # 3-Day Threat Energy (288 steps = 72 hours)
        threat_flux = np.mean(self.grid + 0.35 * self.grid_front, axis=1) # [N]
        df_flux = pd.Series(threat_flux)
        self.energy_3d = df_flux.rolling(window=288, min_periods=1).mean().values.astype(np.float32)[:, np.newaxis]
        self.energy_3d_momentum = np.diff(self.energy_3d, axis=0, prepend=self.energy_3d[0:1, :])

        # Automated Mass Strike Cadence Tracking:
        # Detect mass strikes where >= 16 regions are simultaneously on alert
        mass_mask = (np.sum(self.grid, axis=1) >= 16).astype(np.int32)
        days_since_mass = np.zeros(self.n_steps, dtype=np.float32)
        last_mass_step = -9999
        for s in range(self.n_steps):
            if mass_mask[s] == 1:
                last_mass_step = s
            days_since_mass[s] = min(30.0, (s - last_mass_step) / (4.0 * 24.0))

        # Sigmoid Logistics Replenishment Accumulator: S = 1 / (1 + exp(-0.25 * (days - 14)))
        self.replenish_sigmoid = (1.0 / (1.0 + np.exp(-0.25 * (days_since_mass - 14.0)))).astype(np.float32)[:, np.newaxis]

        # Valid sampling index bounds
        # Kinetic window is 32 steps (8h); Strategic window needs 96 steps (24h)
        self.t_kin_steps = 32
        self.min_t = max(96, self.t_kin_steps)
        self.max_t = self.n_steps - 96 - 1
        self.valid_indices = list(range(self.min_t, self.max_t, sample_stride))
        self.stratified = stratified
        if stratified:
            onsets_source = self.onsets_raion if self.has_raions else self.onsets
            event_idx = [idx for idx in self.valid_indices if np.sum(onsets_source[idx + 1 : idx + 17, :]) > 0]
            calm_idx = [idx for idx in self.valid_indices if np.sum(onsets_source[idx + 1 : idx + 17, :]) == 0]
            if len(event_idx) > 0 and len(calm_idx) > 0:
                rng = np.random.RandomState(42)
                target_size = len(self.valid_indices)
                ev_sampled = rng.choice(event_idx, size=target_size // 2, replace=True)
                cl_sampled = rng.choice(calm_idx, size=target_size // 2, replace=True)
                balanced = []
                for e, c in zip(ev_sampled, cl_sampled):
                    balanced.append(int(e))
                    balanced.append(int(c))
                self.valid_indices = balanced

    def __len__(self) -> int:
        return len(self.valid_indices)

    def get_sample(
        self,
        i: int,
        return_hierarchical: bool = False,
        return_leaf: bool = False,
        return_regional_iso: bool = False,
    ) -> Tuple[np.ndarray, ...]:
        t = self.valid_indices[i]
        K = self.t_kin_steps # 32 steps

        # 1. Kinetic Stream: 32 steps (15m each: t-31 to t) -> [32, 25]
        s_state = self.grid[t - K + 1 : t + 1, :]
        s_front = self.grid_front[t - K + 1 : t + 1, :]
        s_pfwd = self.p_fwd_active[t - K + 1 : t + 1, :]
        s_pbwd = self.p_bwd_active[t - K + 1 : t + 1, :]
        s_pdiff = self.p_fwd_diff[t - K + 1 : t + 1, :]

        # Broadcast static distances: [25, 3] -> [32, 25, 3]
        dist_broad = np.tile(self.static_dist[np.newaxis, :, :], (K, 1, 1)) # [32, 25, 3]
        surge_broad = np.tile(self.surge_flag[t - K + 1 : t + 1, :][:, np.newaxis, :], (1, self.n_regions, 1)) # [32, 25, 1]
        noct_broad = np.tile(self.nocturnal_flag[t - K + 1 : t + 1, :][:, np.newaxis, :], (1, self.n_regions, 1)) # [32, 25, 1]
        # 6 Tactical OSINT Text Flags: [32, 25, 6] -> transpose to [25, 32, 6]
        # [mig31k, ballistics, strat_aviation, shahed, kab, clearance]
        s_text = self.grid_text[t - K + 1 : t + 1, :, :] # [32, 25, 6]
        text_trans = np.transpose(s_text, (1, 0, 2))     # [25, 32, 6]

        # Assemble Kinetic tensor: [25, 32, 16]
        x_kin = np.concatenate([
            np.transpose(s_state, (1, 0))[:, :, np.newaxis],
            np.transpose(s_front, (1, 0))[:, :, np.newaxis],
            np.transpose(s_pfwd, (1, 0))[:, :, np.newaxis],
            np.transpose(s_pbwd, (1, 0))[:, :, np.newaxis],
            np.transpose(s_pdiff, (1, 0))[:, :, np.newaxis],
            np.transpose(dist_broad, (1, 0, 2)),
            np.transpose(surge_broad, (1, 0, 2)),
            np.transpose(noct_broad, (1, 0, 2)),
            text_trans,
        ], axis=-1).astype(np.float32) # [25, 32, 16]

        # 2. Campaign Macro Stream: 24 steps (1h each: sample backwards from t)
        slow_indices = [t - 4 * k for k in range(23, -1, -1)]
        c_state = self.grid[slow_indices, :]
        c_roll7 = self.roll_7d_reg[slow_indices, :]

        c_sin = np.tile(self.sin_h[slow_indices][np.newaxis, :, np.newaxis], (self.n_regions, 1, 1))
        c_cos = np.tile(self.cos_h[slow_indices][np.newaxis, :, np.newaxis], (self.n_regions, 1, 1))
        c_h6 = np.tile(self.harmonic_6h[slow_indices][np.newaxis, :, np.newaxis], (self.n_regions, 1, 1))
        c_nat7 = np.tile(self.roll_7d_nat[slow_indices, 0][np.newaxis, :, np.newaxis], (self.n_regions, 1, 1))
        c_e3d = np.tile(self.energy_3d[slow_indices, 0][np.newaxis, :, np.newaxis], (self.n_regions, 1, 1))
        c_mom = np.tile(self.energy_3d_momentum[slow_indices, 0][np.newaxis, :, np.newaxis], (self.n_regions, 1, 1))
        c_rep = np.tile(self.replenish_sigmoid[slow_indices, 0][np.newaxis, :, np.newaxis], (self.n_regions, 1, 1))
        c_front = np.transpose(self.grid_front[slow_indices, :], (1, 0))[:, :, np.newaxis]

        x_camp = np.concatenate([
            np.transpose(c_state, (1, 0))[:, :, np.newaxis],
            np.transpose(c_roll7, (1, 0))[:, :, np.newaxis],
            c_sin,
            c_cos,
            c_h6,
            c_nat7,
            c_e3d,
            c_mom,
            c_rep,
            c_front
        ], axis=-1).astype(np.float32) # [25, 24, 10]

        # 3. Ground Truth Macro Targets
        # Layer 1: Tactical Onset in Next 1 Hour (4 steps: t+1 to t+4)
        y_onset_1h = np.max(self.onsets[t + 1 : t + 1 + 4, :], axis=0).astype(np.float32)

        # Ground-truth target y_onset_isochrone: onset occurring within each region's physical isochrone window tau(r)
        y_onset_isochrone = np.zeros(self.n_regions, dtype=np.float32)
        for r in range(self.n_regions):
            steps_r = int(self.tau_steps_reg[r])
            y_onset_isochrone[r] = float(np.max(self.onsets[t + 1 : t + 1 + steps_r, r]))

        # Continuous target y_time_to_onset: continuous hours until next siren [0.0, 6.0h]
        y_time_to_onset = np.full(self.n_regions, 6.0, dtype=np.float32)
        future_24_reg = self.onsets[t + 1 : t + 1 + 24, :]
        for r in range(self.n_regions):
            hits_r = np.where(future_24_reg[:, r] > 0)[0]
            if len(hits_r) > 0:
                y_time_to_onset[r] = float((hits_r[0] + 1) * 0.25)

        # Layer 2: Operational Corridor Onset in Next 4 Hours (16 steps: t+1 to t+16)
        y_onset_4h = np.max(self.onsets[t + 1 : t + 1 + 16, :], axis=0).astype(np.float32)

        # Layer 3: Strategic Campaign Intensity (Total alert hours in next 24h: 96 steps)
        y_intensity_24h = (np.sum(self.grid[t + 1 : t + 1 + 96, :], axis=0) * 0.25).astype(np.float32)

        # Idle mask: 1.0 if idle at time t, 0.0 if already on alert
        curr_state = self.grid[t, :]
        idle_mask = (curr_state == 0.0).astype(np.float32)

        if not (return_hierarchical and self.has_raions):
            if return_regional_iso:
                return x_kin, x_camp, y_onset_1h, y_onset_4h, y_intensity_24h, idle_mask, y_onset_isochrone, y_time_to_onset
            return x_kin, x_camp, y_onset_1h, y_onset_4h, y_intensity_24h, idle_mask

        # 4. Micro Raion Stream: 32 steps -> [136, 32, 8]
        s_raion = self.grid_raion[t - K + 1 : t + 1, :]       # [32, 136]
        s_rm_fwd = self.p_micro_fwd_active[t - K + 1 : t + 1, :] # [32, 136]
        s_rm_bwd = self.p_micro_bwd_active[t - K + 1 : t + 1, :] # [32, 136]
        s_rm_diff = self.p_micro_fwd_diff[t - K + 1 : t + 1, :]  # [32, 136]

        d_broad = np.tile((self.d_threat_raion / 1000.0)[np.newaxis, :], (K, 1)) # [32, 136]
        tau_broad = np.tile((self.tau_hours_raion / 6.0)[np.newaxis, :], (K, 1)) # [32, 136]
        surge_r = np.tile(self.surge_flag[t - K + 1 : t + 1, :], (1, NUM_RAIONS)) # [32, 136]
        noct_r = np.tile(self.nocturnal_flag[t - K + 1 : t + 1, :], (1, NUM_RAIONS)) # [32, 136]

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

        # 5. Micro Raion Targets
        # Tactical Raion Onset within individual isochrone tau(d)
        y_raion_iso = np.zeros(NUM_RAIONS, dtype=np.float32)
        for d in range(NUM_RAIONS):
            steps = int(self.tau_steps_raion[d])
            y_raion_iso[d] = float(np.max(self.onsets_raion[t + 1 : t + 1 + steps, d]))

        # Continuous Time-to-Impact [0.0 - 6.0 hours]
        y_tti = np.full(NUM_RAIONS, 6.0, dtype=np.float32)
        future_24 = self.onsets_raion[t + 1 : t + 1 + 24, :] # [24, 136]
        for d in range(NUM_RAIONS):
            hits = np.where(future_24[:, d] > 0)[0]
            if len(hits) > 0:
                y_tti[d] = float((hits[0] + 1) * 0.25)

        idle_mask_raion = (self.grid_raion[t, :] == 0.0).astype(np.float32)

        # 6. Discrete Leaf Stream: 32 steps -> [204, 32, 8]
        if return_leaf and self.has_leaves:
            s_leaf = self.grid_leaf[t - K + 1 : t + 1, :]
            s_lm_fwd = np.matmul(s_rm_fwd, self.m_leaf_raion.T)
            s_lm_bwd = np.matmul(s_rm_bwd, self.m_leaf_raion.T)
            s_lm_diff = np.matmul(s_rm_diff, self.m_leaf_raion.T)

            d_broad_l = np.tile((self.d_threat_leaf / 1000.0)[np.newaxis, :], (K, 1))
            tau_broad_l = np.tile((self.tau_hours_leaf / 6.0)[np.newaxis, :], (K, 1))
            surge_l = np.tile(self.surge_flag[t - K + 1 : t + 1, :], (1, NUM_LEAF_NODES))
            noct_l = np.tile(self.nocturnal_flag[t - K + 1 : t + 1, :], (1, NUM_LEAF_NODES))

            x_leaf = np.stack([
                s_leaf.T,
                s_lm_fwd.T,
                s_lm_bwd.T,
                s_lm_diff.T,
                d_broad_l.T,
                tau_broad_l.T,
                surge_l.T,
                noct_l.T,
            ], axis=-1).astype(np.float32) # [204, 32, 8]

            y_leaf_iso = np.zeros(NUM_LEAF_NODES, dtype=np.float32)
            for l in range(NUM_LEAF_NODES):
                steps_l = int(self.tau_steps_leaf[l])
                y_leaf_iso[l] = float(np.max(self.onsets_leaf[t + 1 : t + 1 + steps_l, l]))

            y_leaf_tti = np.full(NUM_LEAF_NODES, 6.0, dtype=np.float32)
            future_24_leaf = self.onsets_leaf[t + 1 : t + 1 + 24, :]
            for l in range(NUM_LEAF_NODES):
                hits_l = np.where(future_24_leaf[:, l] > 0)[0]
                if len(hits_l) > 0:
                    y_leaf_tti[l] = float((hits_l[0] + 1) * 0.25)

            idle_mask_leaf = (self.grid_leaf[t, :] == 0.0).astype(np.float32)

            if return_regional_iso:
                return (
                    x_kin, x_camp, x_raion, x_leaf,
                    y_onset_1h, y_onset_4h, y_intensity_24h, idle_mask,
                    y_raion_iso, y_tti, idle_mask_raion,
                    y_leaf_iso, y_leaf_tti, idle_mask_leaf,
                    y_onset_isochrone, y_time_to_onset
                )
            return (
                x_kin, x_camp, x_raion, x_leaf,
                y_onset_1h, y_onset_4h, y_intensity_24h, idle_mask,
                y_raion_iso, y_tti, idle_mask_raion,
                y_leaf_iso, y_leaf_tti, idle_mask_leaf
            )

        if return_regional_iso:
            return (
                x_kin, x_camp, x_raion,
                y_onset_1h, y_onset_4h, y_intensity_24h, idle_mask,
                y_raion_iso, y_tti, idle_mask_raion,
                y_onset_isochrone, y_time_to_onset
            )
        return (
            x_kin, x_camp, x_raion,
            y_onset_1h, y_onset_4h, y_intensity_24h, idle_mask,
            y_raion_iso, y_tti, idle_mask_raion
        )

    def __getitem__(self, i: int) -> Tuple[np.ndarray, ...]:
        if self.has_leaves:
            return self.get_sample(i, return_hierarchical=True, return_leaf=True)
        elif self.has_raions:
            return self.get_sample(i, return_hierarchical=True, return_leaf=False)
        else:
            return self.get_sample(i, return_hierarchical=False, return_leaf=False)
