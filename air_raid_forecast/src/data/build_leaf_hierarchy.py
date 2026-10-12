import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure UTF-8 stdout handling on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_CURRENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CURRENT_DIR.parents[1] if _CURRENT_DIR.parent.name == "src" else _CURRENT_DIR.parent
for _p in [str(_REPO_ROOT), str(_REPO_ROOT.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from air_raid_forecast.src.data.build_graph import REGIONS, REGION_TO_IDX
from air_raid_forecast.src.data.raion_hierarchy import (
    RAION_SPEC,
    RAION_NAMES,
    RAION_TO_IDX,
    compute_raion_threat_distances,
)

def norm_name(s: str) -> str:
    return str(s).lower().replace("’", "'").replace("`", "'").replace(" район", "").strip()

def build_and_save_leaf_artifacts():
    candidates = [
        _REPO_ROOT / "air_raid_forecast" / "data" / "raw" / "official_data_uk.csv",
        _REPO_ROOT / "data" / "raw" / "official_data_uk.csv",
    ]
    csv_path = None
    for c in candidates:
        if c.exists():
            csv_path = str(c)
            break

    df = pd.read_csv(csv_path)
    h_df = df[df['level'] == 'hromada'][['oblast', 'raion', 'hromada']].dropna().drop_duplicates()
    
    spec_norm = {norm_name(r): (i, r) for i, r in enumerate(RAION_NAMES)}
    d_threats, tau_hours, tau_steps = compute_raion_threat_distances()

    # 133 hromadas
    hromada_leaves = []
    parent_raion_set = set()
    for _, row in h_df.iterrows():
        h_name = row['hromada']
        r_name = row['raion']
        o_name = row['oblast']
        nr = norm_name(r_name)
        r_idx, canonical_r = spec_norm[nr]
        o_idx = REGION_TO_IDX[o_name]
        parent_raion_set.add(r_name)
        lat = float(RAION_SPEC[r_idx][2])
        lon = float(RAION_SPEC[r_idx][3])
        d_th = float(d_threats[r_idx])
        tau_h = float(tau_hours[r_idx])
        tau_s = int(tau_steps[r_idx])
        hromada_leaves.append({
            "name": h_name,
            "type": "hromada",
            "parent_raion": canonical_r,
            "parent_raion_idx": r_idx,
            "parent_oblast": o_name,
            "parent_oblast_idx": o_idx,
            "latitude": lat,
            "longitude": lon,
            "dist_threat_km": d_th,
            "tau_hours": tau_h,
            "tau_steps_15m": tau_s,
        })

    # 71 raions without child hromadas in dataset
    all_df_raions = set(df['raion'].dropna().unique())
    raions_no_hromadas = sorted(list(all_df_raions - parent_raion_set))
    raion_to_oblast = df[['raion', 'oblast']].dropna().drop_duplicates().set_index('raion')['oblast'].to_dict()

    raion_leaves = []
    for r_name in raions_no_hromadas:
        nr = norm_name(r_name)
        r_idx, canonical_r = spec_norm[nr]
        o_name = raion_to_oblast.get(r_name, RAION_SPEC[r_idx][1])
        o_idx = REGION_TO_IDX[o_name]
        lat = float(RAION_SPEC[r_idx][2])
        lon = float(RAION_SPEC[r_idx][3])
        d_th = float(d_threats[r_idx])
        tau_h = float(tau_hours[r_idx])
        tau_s = int(tau_steps[r_idx])
        raion_leaves.append({
            "name": canonical_r,
            "type": "raion",
            "parent_raion": canonical_r,
            "parent_raion_idx": r_idx,
            "parent_oblast": o_name,
            "parent_oblast_idx": o_idx,
            "latitude": lat,
            "longitude": lon,
            "dist_threat_km": d_th,
            "tau_hours": tau_h,
            "tau_steps_15m": tau_s,
        })

    leaves_spec = hromada_leaves + raion_leaves
    for i, leaf in enumerate(leaves_spec):
        leaf["id"] = i

    print(f"Total leaves: {len(leaves_spec)} (133 hromadas + 71 childless raions)")

    # Save to processed directories
    target_dirs = [
        _REPO_ROOT / "air_raid_forecast" / "data" / "processed",
        _REPO_ROOT / "data" / "processed",
    ]
    for td in target_dirs:
        td.mkdir(parents=True, exist_ok=True)
        with open(td / "leaf_nodes.json", "w", encoding="utf-8") as f:
            json.dump(leaves_spec, f, ensure_ascii=False, indent=2)

        # Assignment matrices:
        # M_leaf_raion: [204, 136]
        m_lr = np.zeros((len(leaves_spec), len(RAION_SPEC)), dtype=np.float32)
        m_lo = np.zeros((len(leaves_spec), len(REGIONS)), dtype=np.float32)
        for i, leaf in enumerate(leaves_spec):
            m_lr[i, leaf["parent_raion_idx"]] = 1.0
            m_lo[i, leaf["parent_oblast_idx"]] = 1.0

        np.save(str(td / "leaf_assignment_matrix.npy"), m_lr)
        np.save(str(td / "leaf_to_oblast_matrix.npy"), m_lo)
        print(f"Saved leaf artifacts to {td}")

    return leaves_spec

if __name__ == "__main__":
    build_and_save_leaf_artifacts()
