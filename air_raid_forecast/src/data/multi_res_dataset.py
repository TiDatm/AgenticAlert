"""
Multi-Resolution Dataset Generator with Strict Anti-Leakage Controls.
Generates synchronized dual-stream tensors for all 25 regions:
- Stream Fast: 15-minute intervals (T=8 steps, past 2 hours)
- Stream Slow: 1-hour intervals (T=24 steps, past 24 hours)
- Multi-Horizon Targets: Binary alert occurrence in next 1h, 4h, and 24h
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

# Ensure project root and parent are on sys.path
# Ensure UTF-8 stdout handling on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from air_raid_forecast.src.data.build_graph import REGIONS, REGION_TO_IDX, build_adjacency_matrix


def build_hierarchical_time_grids(
    df: pd.DataFrame,
    freq: str = "15min",
    return_leaf_grid: bool = False
) -> Tuple[Any, ...]:
    """
    Constructs multi-scale synchronized grids with physical de-saturation and hierarchical inheritance:
    1. grid_strat: [N, 25] binary strategic regional siren state
       (declared if: level == 'oblast' OR capital raion is active OR >= 2 independent raions active)
       -> Resolves frontline sub-district saturation (e.g. Nikopol tube artillery isolation).
    2. grid_front: [N, 25] continuous [0.0, 1.0] (tactical sub-district alert density).
    3. grid_raion: [N, 136] binary tactical micro-siren states with inheritance:
        grid_raion[t, d] = local_raion(t, d) | parent_oblast(t, r)
    4. grid_leaf: [N, 204] binary leaf siren states with downward hierarchical propagation:
       grid_leaf[t, l] = local_leaf(t, l) | parent_raion(t, d) | parent_oblast(t, r)
    """
    from air_raid_forecast.src.data.raion_hierarchy import (
        RAION_TO_IDX, NUM_RAIONS, RAION_NAMES, build_cluster_assignment_matrix,
        get_capital_raion_indices, get_raions_by_oblast,
        LEAF_SPEC, NUM_LEAF_NODES, LEAF_TO_IDX, RAION_TO_HROMADAS
    )

    # Normalize input dates
    start_col = "start" if "start" in df.columns else ("started_at" if "started_at" in df.columns else None)
    end_col = "end" if "end" in df.columns else ("finished_at" if "finished_at" in df.columns else None)
    if start_col is None:
        raise ValueError("DataFrame must contain 'start' or 'started_at' column")

    s_series = pd.to_datetime(df[start_col], utc=True)
    t_min = s_series.min().floor("1h")
    valid_ends = pd.to_datetime(df[end_col], utc=True).dropna() if end_col else pd.Series([], dtype="datetime64[ns, UTC]")
    if len(valid_ends) > 0:
        t_max = max(valid_ends.max(), s_series.max())
    else:
        t_max = s_series.max()
    t_max = t_max.ceil("1h")

    time_index = pd.date_range(start=t_min, end=t_max, freq=freq, tz="UTC")
    n_times = len(time_index)
    n_regions = len(REGIONS)

    grid_oblast_raw = np.zeros((n_times, n_regions), dtype=np.float32)
    grid_raion_raw = np.zeros((n_times, NUM_RAIONS), dtype=np.float32)
    grid_hromada_raw = np.zeros((n_times, 133), dtype=np.float32)
    grid_front_raw = np.zeros((n_times, n_regions), dtype=np.float32)

    # Normalize raion names (handle unicode/ASCII apostrophes and 'район' suffix variations)
    raion_norm = {}
    for r_name, r_idx in RAION_TO_IDX.items():
        cleaned = r_name.lower().replace("’", "'").replace("`", "'")
        raion_norm[cleaned] = r_idx
        raion_norm[r_name.lower()] = r_idx
        stripped = cleaned.replace(" район", "").strip()
        raion_norm[stripped] = r_idx

    # Normalize hromada names
    hromada_spec = [x for x in LEAF_SPEC if x.get("type") == "hromada"]
    hromada_norm = {}
    for h_i, h_obj in enumerate(hromada_spec):
        h_name = h_obj["name"]
        cleaned_h = h_name.lower().replace("’", "'").replace("`", "'").strip()
        hromada_norm[cleaned_h] = h_i
        hromada_norm[h_name] = h_i

    has_level = "level" in df.columns
    raions_map = get_raions_by_oblast()

    for _, row in df.iterrows():
        reg_val = row.get("region_title")
        if pd.isna(reg_val) or not str(reg_val).strip():
            reg_val = row.get("oblast")
        region_name = str(reg_val).strip() if pd.notna(reg_val) else ""

        if region_name not in REGION_TO_IDX:
            continue
        r_idx = REGION_TO_IDX[region_name]
        start_t = pd.to_datetime(row[start_col], utc=True)
        end_val = row.get(end_col) if end_col else None
        end_t = pd.to_datetime(end_val, utc=True) if pd.notnull(end_val) else start_t + pd.Timedelta(hours=2)

        idx_start = time_index.searchsorted(start_t, side="left")
        idx_end = time_index.searchsorted(end_t, side="right")
        if idx_start >= idx_end or idx_start >= n_times:
            continue
        idx_end = min(idx_end, n_times)

        lvl = str(row.get("level", "oblast")).lower() if has_level and pd.notna(row.get("level")) else "oblast"
        if lvl == "oblast":
            grid_oblast_raw[idx_start:idx_end, r_idx] = 1.0
        elif lvl == "hromada":
            grid_front_raw[idx_start:idx_end, r_idx] += 0.5
            h_val = row.get("hromada")
            h_raw = str(h_val).strip().lower().replace("’", "'").replace("`", "'") if pd.notna(h_val) else ""
            h_idx = hromada_norm.get(h_raw)
            if h_idx is not None:
                grid_hromada_raw[idx_start:idx_end, h_idx] = 1.0
        else:
            # level == 'raion'
            grid_front_raw[idx_start:idx_end, r_idx] += 1.0
            raion_val = row.get("raion")
            r_raw = str(raion_val).strip().lower().replace("’", "'").replace("`", "'") if pd.notna(raion_val) else ""
            matched_r_idx = raion_norm.get(r_raw)
            if matched_r_idx is None and r_raw:
                matched_r_idx = raion_norm.get(r_raw + " район")
            if matched_r_idx is None and r_raw:
                matched_r_idx = raion_norm.get(r_raw.replace(" район", "").strip())

            if matched_r_idx is not None:
                grid_raion_raw[idx_start:idx_end, matched_r_idx] = 1.0
            elif lvl == "raion" and region_name in raions_map:
                for c_idx in raions_map[region_name]:
                    grid_raion_raw[idx_start:idx_end, c_idx] = 1.0

    # 1. Downward Hierarchical Propagation for 204 Leaf Nodes:
    # Alert_leaf(t, node) = Alert_local_leaf(t, node) | Alert_parent_raion(t, parent_raion(node)) | Alert_parent_oblast(t, parent_oblast(node))
    grid_leaf = np.zeros((n_times, NUM_LEAF_NODES), dtype=np.float32)
    for l_idx, leaf_obj in enumerate(LEAF_SPEC):
        p_raion_idx = leaf_obj["parent_raion_idx"]
        p_obl_idx = leaf_obj["parent_oblast_idx"]
        obl_alert = grid_oblast_raw[:, p_obl_idx]
        raion_alert = grid_raion_raw[:, p_raion_idx]

        if leaf_obj.get("type") == "hromada":
            h_sub_idx = hromada_norm.get(leaf_obj["name"])
            local_alert = grid_hromada_raw[:, h_sub_idx] if h_sub_idx is not None else 0.0
            grid_leaf[:, l_idx] = np.maximum.reduce([local_alert, raion_alert, obl_alert])
        else:
            # Childless raion endpoint
            grid_leaf[:, l_idx] = np.maximum(raion_alert, obl_alert)

    # 2. Hierarchical Alert Inheritance for 136 Raions:
    # Downward propagation only: Alert_leaf = Local | Parent_raion | Parent_oblast.
    # Raion inherits parent oblast: grid_raion[t, d] = local_raion(t, d) | parent_oblast(t, r)
    S = build_cluster_assignment_matrix()
    inherited_macro = np.matmul(grid_oblast_raw, S.T) # [N, 136]
    grid_raion = np.maximum(grid_raion_raw, inherited_macro).astype(np.float32)

    # 3. Multi-District Regional De-Saturation for 25 Oblasts:
    capital_map = get_capital_raion_indices()
    grid_strat = np.zeros((n_times, n_regions), dtype=np.float32)

    for r_idx, reg_name in enumerate(REGIONS):
        has_obl_alert = grid_oblast_raw[:, r_idx] > 0
        cap_indices = capital_map.get(reg_name, [])
        has_cap_alert = (np.max(grid_raion[:, cap_indices], axis=1) > 0) if len(cap_indices) > 0 else False

        sub_indices = raions_map.get(reg_name, [])
        if len(sub_indices) > 1:
            multi_raion_active = np.sum(grid_raion[:, sub_indices], axis=1) >= 2
        else:
            multi_raion_active = False

        # Strategic siren sounds if: official oblast alert OR capital district on alert OR >= 2 raions saturated
        reg_active = has_obl_alert | has_cap_alert | multi_raion_active
        grid_strat[:, r_idx] = reg_active.astype(np.float32)

    # 4. Normalize frontline tactical intensity
    p99 = np.maximum(np.percentile(grid_front_raw, 99, axis=0, keepdims=True), 1.0)
    grid_front = np.clip(grid_front_raw / p99, 0.0, 1.0).astype(np.float32)

    if return_leaf_grid:
        return time_index, grid_strat, grid_front, grid_raion, grid_leaf
    return time_index, grid_strat, grid_front, grid_raion


def build_dual_time_grids(
    df: pd.DataFrame,
    freq: str = "15min"
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """
    Constructs dual-channel synchronized grids with de-saturation.
    Backward-compatible interface returning (time_index, grid_strat, grid_front).
    """
    time_idx, grid_strat, grid_front, _ = build_hierarchical_time_grids(df, freq=freq)
    return time_idx, grid_strat, grid_front


def build_regular_time_grid(
    df: pd.DataFrame,
    freq: str = "15min",
    level: str = "oblast"
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Constructs a regular time grid. By default (level='oblast'), isolates strategic whole-oblast
    alerts to prevent local frontline artillery saturation from contaminating the grid.
    """
    time_idx, grid_strat, grid_front = build_dual_time_grids(df, freq=freq)
    if level == "all":
        # Union of strategic and frontline alerts
        return time_idx, np.maximum(grid_strat, (grid_front > 0.05).astype(np.float32))
    elif level == "frontline":
        return time_idx, grid_front
    return time_idx, grid_strat


class MultiResolutionDataset:
    """
    PyTorch-compatible dataset builder generating (X_fast, X_slow, Targets)
    without data leakage.
    """
    def __init__(
        self,
        grid_15m: np.ndarray,
        time_index_15m: pd.DatetimeIndex,
        adj_matrix: np.ndarray,
        t_fast_steps: int = 8,    # 8 * 15m = 2 hours
        t_slow_steps: int = 24,   # 24 * 1h = 24 hours
        h1_steps: int = 4,        # 4 * 15m = 1 hour
        h4_steps: int = 16,       # 16 * 15m = 4 hours
        h24_steps: int = 96,      # 96 * 15m = 24 hours
    ):
        self.grid_15m = grid_15m
        self.time_index = time_index_15m
        self.adj = adj_matrix
        self.t_fast_steps = t_fast_steps
        self.t_slow_steps = t_slow_steps
        self.h1_steps = h1_steps
        self.h4_steps = h4_steps
        self.h24_steps = h24_steps
        self.n_regions = grid_15m.shape[1]

        # Valid sample indices (must have 24h history and 24h future)
        self.min_idx = 4 * t_slow_steps  # 96 steps
        self.max_idx = len(grid_15m) - h24_steps - 1

    def __len__(self) -> int:
        return max(0, self.max_idx - self.min_idx)

    def get_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts sample at sequence index t = min_idx + idx.
        Returns:
            X_fast: [25, 8, F_fast]
            X_slow: [25, 24, F_slow]
            Y_targets: [25, 3] (P_1h, P_4h, P_24h)
        """
        t = self.min_idx + idx
        curr_time = self.time_index[t]

        # 1. Fast Stream: [t - 7 : t + 1] -> 8 steps
        slice_fast = self.grid_15m[t - self.t_fast_steps + 1 : t + 1, :] # shape: [8, 25]
        # Transpose to [25, 8, 1]
        act_fast = np.transpose(slice_fast, (1, 0))[:, :, np.newaxis]
        
        # Neighbor active ratio: A * x / degree
        deg = np.maximum(np.sum(self.adj, axis=1, keepdims=True), 1.0)
        # neighbor activity over 8 steps: [25, 8]
        neighbor_act = np.matmul(self.adj, np.transpose(slice_fast, (1, 0))) / deg
        neighbor_act = neighbor_act[:, :, np.newaxis]
        
        # Fast features: [25, 8, 2]
        x_fast = np.concatenate([act_fast, neighbor_act], axis=-1).astype(np.float32)

        # 2. Slow Stream: 24 steps of 1-hour resolution
        # Sample every 4th step backwards from t
        slow_indices = [t - 4 * k for k in range(self.t_slow_steps - 1, -1, -1)]
        slice_slow = self.grid_15m[slow_indices, :] # [24, 25]
        act_slow = np.transpose(slice_slow, (1, 0))[:, :, np.newaxis]

        # Diurnal solar cycle (sin/cos of hour)
        hours = np.array([self.time_index[si].hour + self.time_index[si].minute / 60.0 for si in slow_indices])
        sin_h = np.sin(2 * np.pi * hours / 24.0).astype(np.float32)
        cos_h = np.cos(2 * np.pi * hours / 24.0).astype(np.float32)
        
        # Broadcast diurnal features across all 25 regions: [25, 24, 1]
        sin_h_mat = np.tile(sin_h[np.newaxis, :, np.newaxis], (self.n_regions, 1, 1))
        cos_h_mat = np.tile(cos_h[np.newaxis, :, np.newaxis], (self.n_regions, 1, 1))

        # Slow features: [25, 24, 3]
        x_slow = np.concatenate([act_slow, sin_h_mat, cos_h_mat], axis=-1).astype(np.float32)

        # 3. Targets: Future alert occurrence in (t, t+H]
        # Next 1h: [t+1 : t+1+4]
        y_1h = (np.max(self.grid_15m[t + 1 : t + 1 + self.h1_steps, :], axis=0) > 0).astype(np.float32)
        # Next 4h: [t+1 : t+1+16]
        y_4h = (np.max(self.grid_15m[t + 1 : t + 1 + self.h4_steps, :], axis=0) > 0).astype(np.float32)
        # Next 24h: [t+1 : t+1+96]
        y_24h = (np.max(self.grid_15m[t + 1 : t + 1 + self.h24_steps, :], axis=0) > 0).astype(np.float32)

        y_targets = np.stack([y_1h, y_4h, y_24h], axis=-1) # [25, 3]

        return x_fast, x_slow, y_targets


if __name__ == "__main__":
    print("MultiResolutionDataset module loaded.")
