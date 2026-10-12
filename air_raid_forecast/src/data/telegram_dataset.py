"""
Air Force & OSINT Tactical Telegram Indicator Ingestion & Discretization.
Extracts 6 tactical threat indicators:
[flag_mig31k, flag_ballistics, flag_strat_aviation, flag_shahed_group, flag_kab_tactical, flag_clearance]
aligned into synchronized 15-minute grids [N_steps, 25, 6] with rolling exponential decay
under strict temporal causality (zero future leakage).
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

# UTF-8 stdout configuration for Windows compatibility
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from air_raid_forecast.src.data.build_graph import (
    REGIONS,
    REGION_TO_IDX,
    BORDER_REGIONS,
    FRONTLINE_REGIONS,
)
from air_raid_forecast.src.data.text_features import (
    parse_tactical_text,
    extract_threat_corridor,
    TACTICAL_FEATURE_NAMES,
)

def resolve_telegram_path(file_path: Optional[str] = None) -> Optional[str]:
    """Resolves raw Telegram jsonl path robustly across workspace configurations."""
    if file_path and os.path.exists(file_path):
        return os.path.abspath(file_path)

    candidates = [
        file_path,
        str(_PROJECT_ROOT / "data" / "raw" / "telegram_air_force_alerts.jsonl"),
        str(_PROJECT_ROOT / "air_raid_forecast" / "data" / "raw" / "telegram_air_force_alerts.jsonl"),
        str(_PROJECT_ROOT.parent / "air_raid_forecast" / "data" / "raw" / "telegram_air_force_alerts.jsonl"),
        os.path.join("air_raid_forecast", "data", "raw", "telegram_air_force_alerts.jsonl"),
        os.path.join("data", "raw", "telegram_air_force_alerts.jsonl"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


def load_cached_telegram_messages(file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads raw Telegram messages from JSONL file if available.
    Each line format: {"timestamp": "2024-01-01T12:00:00Z", "text": "..."}
    """
    resolved_path = resolve_telegram_path(file_path)
    if not resolved_path:
        return []

    path = resolved_path


    messages = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    messages.append(item)
    except Exception as e:
        print(f"Warning: Failed to load Telegram messages from {path}: {e}")
        return []

    return messages


def synthesize_deterministic_tactical_grid(
    time_index: pd.DatetimeIndex,
    grid_strat: np.ndarray,
    grid_front: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Deterministic historical fallback extractor:
    Extracts physical OSINT indicators directly from observed air defense dynamics
    without future leakage:
    1. flag_mig31k: Simultaneous mass onset across >= 16 regions (national alert signature)
    2. flag_ballistics: Sudden border onset in Kharkiv/Sumy/Donetsk with 0 rear alerts
    3. flag_strat_aviation: Sustained multi-region cruise missile alert campaigns
    4. flag_shahed_group: Nocturnal (21:00 - 06:00) sequential corridor alert vectors
    5. flag_kab_tactical: Local border/frontline sub-district alert bursts (level: raion/hromada)
    6. flag_clearance: Region-level transitions from 1 -> 0
    Returns raw indicator tensor [N, 25, 6].
    """
    n_steps, n_regions = grid_strat.shape
    raw_tensor = np.zeros((n_steps, n_regions, 6), dtype=np.float32)

    if grid_front is None:
        grid_front = np.zeros_like(grid_strat)

    diffs = np.diff(grid_strat, axis=0, prepend=grid_strat[0:1, :])
    onsets = (diffs > 0).astype(np.float32)      # [N, 25] 1 on calm -> alert
    clearances = (diffs < 0).astype(np.float32)  # [N, 25] 1 on alert -> calm

    # Region index sets
    border_indices = [REGION_TO_IDX[r] for r in BORDER_REGIONS if r in REGION_TO_IDX]
    frontline_indices = [REGION_TO_IDX[r] for r in FRONTLINE_REGIONS if r in REGION_TO_IDX]
    west_indices = [
        REGION_TO_IDX[r]
        for r in [
            "Львівська область", "Волинська область", "Рівненська область",
            "Тернопільська область", "Івано-Франківська область", "Закарпатська область"
        ]
        if r in REGION_TO_IDX
    ]

    hours = np.array([t.hour for t in time_index])
    is_night = (hours >= 21) | (hours <= 6)

    for t in range(n_steps):
        nat_onsets = np.sum(onsets[t, :])
        nat_active = np.sum(grid_strat[t, :])

        # 1. MiG-31K: National simultaneous onset (>= 16 regions activate together)
        if nat_onsets >= 16 or (nat_active >= 22 and np.mean(grid_strat[max(0, t - 4) : t, :]) < 0.3):
            raw_tensor[t, :, 0] = 1.0

        # 2. Ballistics: Sudden border onset in Kharkiv/Sumy/Donetsk/Zaporizhzhia without prior west alerts
        west_active = np.sum(grid_strat[t, west_indices]) if west_indices else 0
        if west_active == 0:
            for b_idx in border_indices:
                if onsets[t, b_idx] == 1.0:
                    raw_tensor[t, b_idx, 1] = 1.0
                    # Border neighbors receive immediate ballistic pre-warning
                    for neighbor_idx in [REGION_TO_IDX.get("Полтавська область"), REGION_TO_IDX.get("Дніпропетровська область")]:
                        if neighbor_idx is not None:
                            raw_tensor[t, neighbor_idx, 1] = 0.5

        # 3. Strategic Aviation: Prolonged nationwide cruise missile strikes (>= 10 active regions)
        if nat_active >= 12 and not is_night[t]:
            raw_tensor[t, :, 2] = 0.8

        # 4. Shahed UAV Swarm: Nighttime alerts along transit corridors + downwind corridor pre-warning
        if is_night[t]:
            active_drone_nodes = [r_idx for r_idx in range(n_regions) if (onsets[t, r_idx] == 1.0 or grid_strat[t, r_idx] == 1.0)]
            for r_idx in active_drone_nodes:
                raw_tensor[t, r_idx, 3] = 1.0

            # Forward transit corridor propagation:
            # If eastern/southern ingress nodes are active, project flight vector along Dnipro / Western corridor
            has_east_ingress = any(grid_strat[t, b_idx] == 1.0 for b_idx in border_indices)
            if has_east_ingress:
                # Central transit corridor receives early notice
                for central_name in ["Полтавська область", "Черкаська область", "Кіровоградська область", "Київська область", "м. Київ"]:
                    c_idx = REGION_TO_IDX.get(central_name)
                    if c_idx is not None and grid_strat[t, c_idx] == 0:
                        raw_tensor[t, c_idx, 3] = max(raw_tensor[t, c_idx, 3], 0.65)

                # Deep rear corridor receives kinematic early-warning notice
                for rear_name in ["Вінницька область", "Житомирська область", "Хмельницька область", "Рівненська область", "Львівська область", "Тернопільська область", "Волинська область"]:
                    w_idx = REGION_TO_IDX.get(rear_name)
                    if w_idx is not None and grid_strat[t, w_idx] == 0:
                        raw_tensor[t, w_idx, 3] = max(raw_tensor[t, w_idx, 3], 0.40)

        # 5. KAB Tactical Aviation: Frontline/border sub-district alert activity
        for f_idx in frontline_indices + border_indices:
            if grid_front[t, f_idx] > 0.15 or (onsets[t, f_idx] == 1.0 and not is_night[t]):
                raw_tensor[t, f_idx, 4] = 1.0


        # 6. Clearance: Regional siren drop from 1 to 0
        for r_idx in range(n_regions):
            if clearances[t, r_idx] == 1.0:
                raw_tensor[t, r_idx, 5] = 1.0

    return raw_tensor


def build_tactical_text_grid(
    time_index: pd.DatetimeIndex,
    grid_strat: Optional[np.ndarray] = None,
    grid_front: Optional[np.ndarray] = None,
    decay_factor: float = 0.50,
) -> np.ndarray:
    """
    Builds synchronized [N_steps, 25, 6] tactical OSINT signal grid.
    Applies strict temporal causality (no future leakage) and rolling exponential decay:
    - Indicators: [mig31k, ballistics, strat_aviation, shahed_group, kab_tactical, clearance]
    - S(t) = max(Raw(t), S(t-1) * decay_factor)
    - On clearance (flag_clearance == 1.0), threat indicators are immediately cleared.
    """
    n_steps = len(time_index)
    n_regions = len(REGIONS)
    raw_grid = np.zeros((n_steps, n_regions, 6), dtype=np.float32)

    # 1. Attempt ingestion from cached raw Telegram messages
    messages = load_cached_telegram_messages()
    has_raw_messages = len(messages) > 0

    if has_raw_messages:
        for msg in messages:
            ts_str = msg.get("timestamp")
            text = msg.get("text", "")
            if not ts_str or not text:
                continue

            try:
                t_msg = pd.to_datetime(ts_str, utc=True)
            except Exception:
                continue

            # Strict causality: message at t_msg is first visible at bin_idx >= t_msg
            bin_idx = time_index.searchsorted(t_msg, side="left")
            if bin_idx >= n_steps:
                continue

            flags = parse_tactical_text(text)
            corridor = extract_threat_corridor(text)
            mask = corridor["region_mask"] # [25]

            # Ingest 6 flags
            flag_vals = [
                flags["flag_mig31k"],
                flags["flag_ballistics"],
                flags["flag_strat_aviation"],
                flags["flag_shahed_group"],
                flags["flag_kab_tactical"],
                flags["flag_clearance"],
            ]

            for c, f_val in enumerate(flag_vals):
                if f_val > 0.0:
                    for r_idx in range(n_regions):
                        if mask[r_idx] > 0.0 or c == 0:  # MiG-31K is nationwide
                            raw_grid[bin_idx, r_idx, c] = max(raw_grid[bin_idx, r_idx, c], f_val)

    # 2. If messages are unavailable or sparse, fuse with deterministic extraction
    if not has_raw_messages or np.sum(raw_grid) < 10.0:
        if grid_strat is not None:
            synth_grid = synthesize_deterministic_tactical_grid(time_index, grid_strat, grid_front)
            raw_grid = np.maximum(raw_grid, synth_grid)

    # 3. Apply Causal Rolling Exponential Decay (Strictly causal: S[t] depends only on S[t-1] and raw[t])
    grid_out = np.zeros((n_steps, n_regions, 6), dtype=np.float32)
    grid_out[0, :, :] = raw_grid[0, :, :]

    for t in range(1, n_steps):
        for r in range(n_regions):
            is_clear = (raw_grid[t, r, 5] == 1.0)
            if is_clear:
                # Immediate cancellation of threat flags
                grid_out[t, r, 0:5] = 0.0
                grid_out[t, r, 5] = max(1.0, grid_out[t - 1, r, 5] * decay_factor)
            else:
                for c in range(6):
                    grid_out[t, r, c] = max(raw_grid[t, r, c], grid_out[t - 1, r, c] * decay_factor)

    return grid_out.astype(np.float32)


if __name__ == "__main__":
    print("Testing telegram_dataset module...")
    times = pd.date_range("2024-01-01 00:00:00", periods=96, freq="15min", tz="UTC")
    dummy_strat = np.zeros((96, 25), dtype=np.float32)
    # Simulate a sudden border alert at t=10 in Kharkiv (idx 19)
    dummy_strat[10:14, 19] = 1.0

    tactical_grid = build_tactical_text_grid(times, grid_strat=dummy_strat)
    print(f"Tactical grid generated with shape: {tactical_grid.shape}")
    print(f"Kharkiv ballistic indicator at t=10: {tactical_grid[10, 19, 1]:.2f}")
    print(f"Kharkiv ballistic indicator at t=11 (decayed): {tactical_grid[11, 19, 1]:.2f}")
    print(f"Kharkiv ballistic indicator at t=12 (decayed): {tactical_grid[12, 19, 1]:.2f}")
    assert tactical_grid.shape == (96, 25, 6)
    assert tactical_grid[10, 19, 1] > 0.0
    assert tactical_grid[11, 19, 1] < tactical_grid[10, 19, 1]
    print("[+] All telegram_dataset invariant checks passed successfully!")
