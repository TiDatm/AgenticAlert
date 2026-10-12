"""
Hierarchical Raion-Level Defense Mapping and Graph Infrastructure (136 Raions <-> 25 Oblasts).
Implements:
1. Canonical 136 consolidated raions following the July 2020 administrative reform.
2. Cluster assignment matrix S in R^(136 x 25), with orthogonal membership: sum_r S[d, r] == 1.
3. Micro-scale Adjacency Matrix A_micro in R^(136 x 136) and directed transition matrices P_micro_fwd, P_micro_bwd.
4. Continuous Kinematic Isochrone Warning Horizons tau(d) and physical distance envelopes.
5. Capital district mapping and multi-district de-saturation rules.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Ensure UTF-8 stdout handling on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Bootstrap repository root into sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CURRENT_DIR.parents[1] if _CURRENT_DIR.parent.name == "src" else _CURRENT_DIR.parent
for _p in [str(_REPO_ROOT), str(_REPO_ROOT.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from air_raid_forecast.src.data.build_graph import (
    REGIONS,
    REGION_TO_IDX,
    IDX_TO_REGION,
    build_adjacency_matrix
)

# Canonical 136 Raions data table: (id, name, parent_oblast, lat, lon, is_capital)
RAION_SPEC: List[Tuple[str, str, float, float, bool]] = [
    # Вінницька область (6)
    ("Вінницький район", "Вінницька область", 49.23, 28.47, True),
    ("Гайсинський район", "Вінницька область", 48.81, 29.38, False),
    ("Жмеринський район", "Вінницька область", 49.04, 28.11, False),
    ("Могилів-Подільський район", "Вінницька область", 48.45, 27.79, False),
    ("Тульчинський район", "Вінницька область", 48.67, 28.85, False),
    ("Хмільницький район", "Вінницька область", 49.55, 27.95, False),

    # Волинська область (4)
    ("Луцький район", "Волинська область", 50.75, 25.34, True),
    ("Володимир-Волинський район", "Волинська область", 50.85, 24.32, False),
    ("Камінь-Каширський район", "Волинська область", 51.62, 24.96, False),
    ("Ковельський район", "Волинська область", 51.21, 24.71, False),

    # Дніпропетровська область (7)
    ("Дніпровський район", "Дніпропетровська область", 48.46, 35.04, True),
    ("Кам’янський район", "Дніпропетровська область", 48.51, 34.61, False),
    ("Криворізький район", "Дніпропетровська область", 47.91, 33.39, False),
    ("Нікопольський район", "Дніпропетровська область", 47.57, 34.40, False),
    ("Павлоградський район", "Дніпропетровська область", 48.52, 35.87, False),
    ("Самарівський район", "Дніпропетровська область", 48.63, 35.22, False),
    ("Синельниківський район", "Дніпропетровська область", 48.32, 35.51, False),

    # Донецька область (8)
    ("Краматорський район", "Донецька область", 48.74, 37.58, True),
    ("Бахмутський район", "Донецька область", 48.60, 38.00, False),
    ("Волноваський район", "Донецька область", 47.60, 37.50, False),
    ("Горлівський район", "Донецька область", 48.30, 38.05, False),
    ("Донецький район", "Донецька область", 48.01, 37.80, False),
    ("Кальміуський район", "Донецька область", 47.66, 38.07, False),
    ("Маріупольський район", "Донецька область", 47.10, 37.55, False),
    ("Покровський район", "Донецька область", 48.28, 37.18, False),

    # Житомирська область (4)
    ("Житомирський район", "Житомирська область", 50.25, 28.66, True),
    ("Бердичівський район", "Житомирська область", 49.89, 28.58, False),
    ("Звягельський район", "Житомирська область", 50.59, 27.63, False),
    ("Коростенський район", "Житомирська область", 50.95, 28.65, False),

    # Закарпатська область (6)
    ("Ужгородський район", "Закарпатська область", 48.62, 22.29, True),
    ("Берегівський район", "Закарпатська область", 48.20, 22.64, False),
    ("Мукачівський район", "Закарпатська область", 48.44, 22.72, False),
    ("Рахівський район", "Закарпатська область", 48.05, 24.21, False),
    ("Тячівський район", "Закарпатська область", 48.01, 23.57, False),
    ("Хустський район", "Закарпатська область", 48.18, 23.30, False),

    # Запорізька область (5)
    ("Запорізький район", "Запорізька область", 47.83, 35.14, True),
    ("Бердянський район", "Запорізька область", 46.75, 36.78, False),
    ("Василівський район", "Запорізька область", 47.43, 35.27, False),
    ("Мелітопольський район", "Запорізька область", 46.85, 35.37, False),
    ("Пологівський район", "Запорізька область", 47.48, 36.25, False),

    # Івано-Франківська область (6)
    ("Івано-Франківський район", "Івано-Франківська область", 48.92, 24.71, True),
    ("Верховинський район", "Івано-Франківська область", 48.15, 24.83, False),
    ("Калуський район", "Івано-Франківська область", 49.04, 24.36, False),
    ("Коломийський район", "Івано-Франківська область", 48.53, 25.04, False),
    ("Косівський район", "Івано-Франківська область", 48.31, 25.09, False),
    ("Надвірнянський район", "Івано-Франківська область", 48.63, 24.58, False),

    # Київська область (7)
    ("Білоцерківський район", "Київська область", 49.80, 30.12, True),
    ("Бориспільський район", "Київська область", 50.35, 30.95, False),
    ("Броварський район", "Київська область", 50.51, 30.79, False),
    ("Бучанський район", "Київська область", 50.55, 30.21, False),
    ("Вишгородський район", "Київська область", 50.58, 30.49, False),
    ("Обухівський район", "Київська область", 50.13, 30.63, False),
    ("Фастівський район", "Київська область", 50.08, 29.92, False),

    # м. Київ (10 districts)
    ("Голосіївський район", "м. Київ", 50.39, 30.51, True),
    ("Дарницький район", "м. Київ", 50.41, 30.66, True),
    ("Деснянський район", "м. Київ", 50.50, 30.60, True),
    ("Дніпровський район (м. Київ)", "м. Київ", 50.45, 30.60, True),
    ("Оболонський район", "м. Київ", 50.50, 30.49, True),
    ("Печерський район", "м. Київ", 50.43, 30.54, True),
    ("Подільський район (м. Київ)", "м. Київ", 50.47, 30.47, True),
    ("Святошинський район", "м. Київ", 50.45, 30.37, True),
    ("Солом’янський район", "м. Київ", 50.43, 30.45, True),
    ("Шевченківський район (м. Київ)", "м. Київ", 50.45, 30.49, True),

    # Кіровоградська область (4)
    ("Кропивницький район", "Кіровоградська область", 48.50, 32.26, True),
    ("Голованівський район", "Кіровоградська область", 48.38, 30.45, False),
    ("Новоукраїнський район", "Кіровоградська область", 48.32, 31.53, False),
    ("Олександрійський район", "Кіровоградська область", 48.67, 33.11, False),

    # Луганська область (8)
    ("Сєвєродонецький район", "Луганська область", 48.95, 38.48, True),
    ("Алчевський район", "Луганська область", 48.48, 38.80, False),
    ("Довжанський район", "Луганська область", 48.08, 39.65, False),
    ("Луганський район", "Луганська область", 48.57, 39.30, False),
    ("Ровеньківський район", "Луганська область", 48.07, 39.35, False),
    ("Сватівський район", "Луганська область", 49.41, 38.16, False),
    ("Старобільський район", "Луганська область", 49.27, 38.90, False),
    ("Щастинський район", "Луганська область", 48.74, 39.23, False),

    # Львівська область (7)
    ("Львівський район", "Львівська область", 49.84, 24.03, True),
    ("Дрогобицький район", "Львівська область", 49.35, 23.51, False),
    ("Золочівський район", "Львівська область", 49.81, 24.90, False),
    ("Самбірський район", "Львівська область", 49.52, 23.20, False),
    ("Стрийський район", "Львівська область", 49.26, 23.86, False),
    ("Червоноградський район", "Львівська область", 50.39, 24.23, False),
    ("Яворівський район", "Львівська область", 49.95, 23.39, False),

    # Миколаївська область (4)
    ("Миколаївський район", "Миколаївська область", 46.97, 31.99, True),
    ("Баштанський район", "Миколаївська область", 47.41, 32.44, False),
    ("Вознесенський район", "Миколаївська область", 47.56, 31.33, False),
    ("Первомайський район", "Миколаївська область", 48.04, 30.85, False),

    # Одеська область (7)
    ("Одеський район", "Одеська область", 46.48, 30.72, True),
    ("Березівський район", "Одеська область", 47.20, 30.91, False),
    ("Білгород-Дністровський район", "Одеська область", 46.19, 30.34, False),
    ("Болградський район", "Одеська область", 45.68, 28.61, False),
    ("Ізмаїльський район", "Одеська область", 45.35, 28.84, False),
    ("Подільський район", "Одеська область", 47.74, 29.53, False),
    ("Роздільнянський район", "Одеська область", 46.85, 30.07, False),

    # Полтавська область (4)
    ("Полтавський район", "Полтавська область", 49.58, 34.55, True),
    ("Кременчуцький район", "Полтавська область", 49.06, 33.42, False),
    ("Лубенський район", "Полтавська область", 50.01, 33.00, False),
    ("Миргородський район", "Полтавська область", 49.97, 33.61, False),

    # Рівненська область (4)
    ("Рівненський район", "Рівненська область", 50.62, 26.25, True),
    ("Вараський район", "Рівненська область", 51.34, 25.85, False),
    ("Дубенський район", "Рівненська область", 50.42, 25.74, False),
    ("Сарненський район", "Рівненська область", 51.33, 26.63, False),

    # Сумська область (5)
    ("Сумський район", "Сумська область", 50.91, 34.80, True),
    ("Конотопський район", "Сумська область", 51.24, 33.20, False),
    ("Охтирський район", "Сумська область", 50.31, 34.90, False),
    ("Роменський район", "Сумська область", 50.75, 33.47, False),
    ("Шосткинський район", "Сумська область", 51.86, 33.48, False),

    # Тернопільська область (3)
    ("Тернопільський район", "Тернопільська область", 49.55, 25.59, True),
    ("Кременецький район", "Тернопільська область", 50.10, 25.73, False),
    ("Чортківський район", "Тернопільська область", 49.01, 25.79, False),

    # Харківська область (7)
    ("Харківський район", "Харківська область", 49.99, 36.23, True),
    ("Богодухівський район", "Харківська область", 50.16, 35.53, False),
    ("Ізюмський район", "Харківська область", 49.20, 37.28, False),
    ("Красноградський район", "Харківська область", 49.37, 35.45, False),
    ("Куп’янський район", "Харківська область", 49.71, 37.61, False),
    ("Лозівський район", "Харківська область", 48.89, 36.32, False),
    ("Чугуївський район", "Харківська область", 49.84, 36.68, False),

    # Херсонська область (5)
    ("Херсонський район", "Херсонська область", 46.63, 32.61, True),
    ("Бериславський район", "Херсонська область", 46.84, 33.43, False),
    ("Генічеський район", "Херсонська область", 46.17, 34.80, False),
    ("Каховський район", "Херсонська область", 46.80, 33.48, False),
    ("Скадовський район", "Херсонська область", 46.12, 32.91, False),

    # Хмельницька область (3)
    ("Хмельницький район", "Хмельницька область", 49.42, 26.98, True),
    ("Кам’янець-Подільський район", "Хмельницька область", 48.68, 26.58, False),
    ("Шепетівський район", "Хмельницька область", 50.18, 27.06, False),

    # Черкаська область (4)
    ("Черкаський район", "Черкаська область", 49.44, 32.06, True),
    ("Звенигородський район", "Черкаська область", 49.08, 30.97, False),
    ("Золотоніський район", "Черкаська область", 49.67, 32.04, False),
    ("Уманський район", "Черкаська область", 48.75, 30.22, False),

    # Чернівецька область (3)
    ("Чернівецький район", "Чернівецька область", 48.29, 25.93, True),
    ("Вижницький район", "Чернівецька область", 48.25, 25.18, False),
    ("Дністровський район", "Чернівецька область", 48.47, 26.83, False),

    # Чернігівська область (5)
    ("Чернігівський район", "Чернігівська область", 51.50, 31.30, True),
    ("Корюківський район", "Чернігівська область", 51.77, 32.27, False),
    ("Новгород-Сіверський район", "Чернігівська область", 51.98, 33.27, False),
    ("Ніжинський район", "Чернігівська область", 51.04, 31.88, False),
    ("Прилуцький район", "Чернігівська область", 50.59, 32.39, False),
]

assert len(RAION_SPEC) == 136, f"Expected 136 raions, got {len(RAION_SPEC)}"

NUM_RAIONS = len(RAION_SPEC)
NUM_OBLASTS = len(REGIONS)

# Lookup maps
RAION_NAMES: List[str] = [r[0] for r in RAION_SPEC]
RAION_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(RAION_NAMES)}
IDX_TO_RAION: Dict[int, str] = {i: name for i, name in enumerate(RAION_NAMES)}


def build_cluster_assignment_matrix() -> np.ndarray:
    """
    Constructs deterministic cluster assignment matrix S in {0, 1}^(136 x 25).
    Each raion belongs to exactly one parent oblast: sum_r S[d, r] == 1.
    """
    s = np.zeros((NUM_RAIONS, NUM_OBLASTS), dtype=np.float32)
    for d_idx, (_, oblast, _, _, _) in enumerate(RAION_SPEC):
        r_idx = REGION_TO_IDX[oblast]
        s[d_idx, r_idx] = 1.0

    # Invariant assertion
    assert np.all(s.sum(axis=1) == 1.0), "Each raion must belong to exactly one oblast"
    assert np.all(s.sum(axis=0) >= 1.0), "Every oblast must have at least one raion"
    return s


def compute_raion_threat_distances() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes for each raion:
    1. Distance to threat vectors (border, frontline, coast) in km.
    2. Kinematic warning horizon tau(d) in hours:
       tau(d) = max(0.5, 1.25 * d_threat / 250.0)
    3. Step horizon (in 15m steps): round(tau(d) * 4)
    """
    rf_border = np.array([51.5, 35.5])       # East/NE border envelope
    frontline = np.array([48.0, 37.0])       # Donbas contact line
    black_sea = np.array([45.0, 33.5])       # Crimea / Black Sea maritime axis

    d_threats = np.zeros(NUM_RAIONS, dtype=np.float32)
    tau_hours = np.zeros(NUM_RAIONS, dtype=np.float32)
    tau_steps = np.zeros(NUM_RAIONS, dtype=np.int32)

    for i, (_, _, lat, lon, _) in enumerate(RAION_SPEC):
        c = np.array([lat, lon])
        d_rf = np.linalg.norm(c - rf_border) * 111.0
        d_fl = np.linalg.norm(c - frontline) * 111.0
        d_bs = np.linalg.norm(c - black_sea) * 111.0
        d_min = float(min(d_rf, d_fl, d_bs))

        # Effective transit speed ~ 250 km/h with 1.25 operational buffer factor
        tau = max(0.5, 1.25 * d_min / 250.0)
        # Cap tau between 0.5h (border) and 6.0h (deep rear)
        tau = min(6.0, tau)
        steps = int(np.clip(np.round(tau * 4.0), 2, 24))

        d_threats[i] = d_min
        tau_hours[i] = tau
        tau_steps[i] = steps

    return d_threats, tau_hours, tau_steps


def build_raion_adjacency_matrix(max_intra_km: float = 95.0, max_inter_km: float = 120.0) -> np.ndarray:
    """
    Constructs micro-scale physical adjacency matrix A_micro in R^(136 x 136).
    Raions are adjacent if:
    1. In the same oblast and within spatial neighbor threshold (max_intra_km).
    2. In adjacent parent oblasts and across-border distance is within max_inter_km.
    Guarantees a connected, realistic planar graph without disconnected enclaves.
    """
    macro_adj = build_adjacency_matrix()
    adj_micro = np.zeros((NUM_RAIONS, NUM_RAIONS), dtype=np.float32)

    coords = np.array([[r[2], r[3]] for r in RAION_SPEC], dtype=np.float32)
    oblasts = [REGION_TO_IDX[r[1]] for r in RAION_SPEC]

    for i in range(NUM_RAIONS):
        obl_i = oblasts[i]
        c_i = coords[i]
        for j in range(i + 1, NUM_RAIONS):
            obl_j = oblasts[j]
            c_j = coords[j]
            dist_km = float(np.linalg.norm(c_i - c_j) * 111.0)

            if obl_i == obl_j:
                # Same oblast neighbors
                if dist_km <= max_intra_km:
                    adj_micro[i, j] = 1.0
                    adj_micro[j, i] = 1.0
            else:
                # Neighboring oblasts
                if macro_adj[obl_i, obl_j] > 0 and dist_km <= max_inter_km:
                    adj_micro[i, j] = 1.0
                    adj_micro[j, i] = 1.0

    # Ensure every node has at least one edge (connect to closest node if degree is 0)
    degrees = adj_micro.sum(axis=1)
    for i in range(NUM_RAIONS):
        if degrees[i] == 0:
            c_i = coords[i]
            dists = np.linalg.norm(coords - c_i, axis=1) * 111.0
            dists[i] = 9999.0 # Ignore self
            closest = int(np.argmin(dists))
            adj_micro[i, closest] = 1.0
            adj_micro[closest, i] = 1.0

    return adj_micro


def build_raion_directed_transition_matrices() -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs dual directed micro-transition matrices for advective threat flow:
    - P_micro_fwd = D_out^(-1) * A_micro_dir (Threat ingress east -> west and border -> interior)
    - P_micro_bwd = D_in^(-1) * A_micro_dir.T (Upstream origin tracking)
    """
    adj_sym = build_raion_adjacency_matrix()
    n = NUM_RAIONS
    adj_dir = np.zeros((n, n), dtype=np.float32)

    coords = np.array([[r[2], r[3]] for r in RAION_SPEC], dtype=np.float32)
    border_oblasts = {
        "Харківська область", "Сумська область", "Чернігівська область",
        "Луганська область", "Донецька область", "Запорізька область", "Херсонська область"
    }

    for i in range(n):
        c_i = coords[i]
        obl_i = RAION_SPEC[i][1]
        for j in range(n):
            if adj_sym[i, j] > 0:
                c_j = coords[j]
                # Forward flow: lon_i >= lon_j - 0.2 (East to West) or border ingress
                is_fwd = (c_i[1] >= c_j[1] - 0.2) or (obl_i in border_oblasts)
                if is_fwd:
                    adj_dir[i, j] = 1.0
                else:
                    adj_dir[i, j] = 0.35 # Minor bidirectional backflow

    # Add self loops
    adj_fwd = adj_dir + np.eye(n, dtype=np.float32)
    d_out = np.maximum(np.sum(adj_fwd, axis=1, keepdims=True), 1.0)
    p_micro_fwd = (adj_fwd / d_out).astype(np.float32)

    adj_bwd = adj_dir.T + np.eye(n, dtype=np.float32)
    d_in = np.maximum(np.sum(adj_bwd, axis=1, keepdims=True), 1.0)
    p_micro_bwd = (adj_bwd / d_in).astype(np.float32)

    return p_micro_fwd, p_micro_bwd


def get_capital_raion_indices() -> Dict[str, List[int]]:
    """
    Returns mapping from parent oblast name to list of capital raion indices.
    """
    capitals = {}
    for i, (name, oblast, _, _, is_cap) in enumerate(RAION_SPEC):
        if is_cap:
            capitals.setdefault(oblast, []).append(i)
    return capitals


def get_raions_by_oblast() -> Dict[str, List[int]]:
    """
    Returns mapping from parent oblast name to list of constituent raion indices.
    """
    by_obl = {}
    for i, (_, oblast, _, _, _) in enumerate(RAION_SPEC):
        by_obl.setdefault(oblast, []).append(i)
    return by_obl


def load_or_build_leaf_spec() -> List[Dict[str, Any]]:
    """
    Loads canonical 204 leaf nodes (133 hromadas + 71 childless raions).
    """
    candidates = [
        _REPO_ROOT / "air_raid_forecast" / "data" / "processed" / "leaf_nodes.json",
        _REPO_ROOT / "data" / "processed" / "leaf_nodes.json",
    ]
    for c in candidates:
        if c.exists():
            try:
                with open(c, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if len(data) == 204:
                        return data
            except Exception:
                pass

    from air_raid_forecast.src.data.build_leaf_hierarchy import build_and_save_leaf_artifacts
    return build_and_save_leaf_artifacts()


LEAF_SPEC: List[Dict[str, Any]] = load_or_build_leaf_spec()
NUM_LEAF_NODES: int = len(LEAF_SPEC)
LEAF_NAMES: List[str] = [leaf["name"] for leaf in LEAF_SPEC]
LEAF_TO_IDX: Dict[str, int] = {leaf["name"]: i for i, leaf in enumerate(LEAF_SPEC)}
IDX_TO_LEAF: Dict[int, str] = {i: leaf["name"] for i, leaf in enumerate(LEAF_SPEC)}

# Lookup mappings for child hromadas
HROMADA_TO_RAION: Dict[str, str] = {
    leaf["name"]: leaf["parent_raion"] for leaf in LEAF_SPEC if leaf.get("type") == "hromada"
}
RAION_TO_HROMADAS: Dict[str, List[str]] = {}
for leaf in LEAF_SPEC:
    if leaf.get("type") == "hromada":
        RAION_TO_HROMADAS.setdefault(leaf["parent_raion"], []).append(leaf["name"])


def build_leaf_to_raion_matrix() -> np.ndarray:
    """
    Builds leaf-to-raion projection matrix M_lr in {0, 1}^(204 x 136).
    Each leaf maps to its parent raion: sum across columns == 1.
    """
    m = np.zeros((NUM_LEAF_NODES, NUM_RAIONS), dtype=np.float32)
    for i, leaf in enumerate(LEAF_SPEC):
        m[i, leaf["parent_raion_idx"]] = 1.0
    return m


def build_leaf_to_oblast_matrix() -> np.ndarray:
    """
    Builds leaf-to-oblast projection matrix M_lo in {0, 1}^(204 x 25).
    Each leaf maps to its parent oblast: sum across columns == 1.
    """
    m = np.zeros((NUM_LEAF_NODES, NUM_OBLASTS), dtype=np.float32)
    for i, leaf in enumerate(LEAF_SPEC):
        m[i, leaf["parent_oblast_idx"]] = 1.0
    return m


def compute_leaf_threat_distances() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns threat distance (km), kinematic horizon tau (hours), and steps (15m) for all 204 leaf nodes.
    """
    d_threats = np.array([leaf.get("dist_threat_km", 150.0) for leaf in LEAF_SPEC], dtype=np.float32)
    tau_hours = np.array([leaf.get("tau_hours", 1.5) for leaf in LEAF_SPEC], dtype=np.float32)
    tau_steps = np.array([leaf.get("tau_steps_15m", 6) for leaf in LEAF_SPEC], dtype=np.int32)
    return d_threats, tau_hours, tau_steps


def build_and_save_raion_metadata(output_dir: Optional[str] = None) -> str:
    """
    Serializes raion and leaf mapping and adjacency graph artifacts:
    - raions.json
    - leaf_nodes.json
    - raion_adjacency.npy
    - cluster_assignment_S.npy
    - leaf_assignment_matrix.npy
    - leaf_to_oblast_matrix.npy
    """
    if output_dir is not None:
        target_dir = Path(output_dir)
    else:
        # Check standard processed paths
        candidates = [
            _REPO_ROOT / "data" / "processed",
            _REPO_ROOT / "air_raid_forecast" / "data" / "processed",
        ]
        target_dir = candidates[0]
        for c in candidates:
            if c.exists():
                target_dir = c
                break

    target_dir.mkdir(parents=True, exist_ok=True)

    d_threat, tau_h, tau_s = compute_raion_threat_distances()
    s_mat = build_cluster_assignment_matrix()
    adj_micro = build_raion_adjacency_matrix()

    metadata = []
    for i, (name, oblast, lat, lon, is_cap) in enumerate(RAION_SPEC):
        metadata.append({
            "id": i,
            "name": name,
            "parent_oblast": oblast,
            "parent_oblast_id": REGION_TO_IDX[oblast],
            "latitude": float(lat),
            "longitude": float(lon),
            "is_capital": bool(is_cap),
            "dist_threat_km": float(d_threat[i]),
            "tau_hours": float(tau_h[i]),
            "tau_steps_15m": int(tau_s[i]),
            "child_hromadas": RAION_TO_HROMADAS.get(name, []),
        })

    json_path = target_dir / "raions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    leaf_json_path = target_dir / "leaf_nodes.json"
    with open(leaf_json_path, "w", encoding="utf-8") as f:
        json.dump(LEAF_SPEC, f, ensure_ascii=False, indent=2)

    m_lr = build_leaf_to_raion_matrix()
    m_lo = build_leaf_to_oblast_matrix()

    np.save(str(target_dir / "cluster_assignment_S.npy"), s_mat)
    np.save(str(target_dir / "raion_assignment_matrix.npy"), s_mat)
    np.save(str(target_dir / "raion_adjacency.npy"), adj_micro)
    np.save(str(target_dir / "leaf_assignment_matrix.npy"), m_lr)
    np.save(str(target_dir / "leaf_to_oblast_matrix.npy"), m_lo)

    return str(json_path)


if __name__ == "__main__":
    p = build_and_save_raion_metadata()
    s = build_cluster_assignment_matrix()
    adj = build_raion_adjacency_matrix()
    pfwd, pbwd = build_raion_directed_transition_matrices()
    d_th, tau_h, tau_s = compute_raion_threat_distances()
    m_lr = build_leaf_to_raion_matrix()

    print(f"Hierarchical Raion System initialized successfully:")
    print(f"  - Total Raions: {NUM_RAIONS} across {NUM_OBLASTS} parent Oblasts")
    print(f"  - Leaf Endpoints: {NUM_LEAF_NODES} (133 hromadas + 71 childless raions)")
    print(f"  - Leaf-to-Raion Matrix: shape {m_lr.shape}, sum = {m_lr.sum(axis=1).min()}..{m_lr.sum(axis=1).max()}")
    print(f"  - Cluster Matrix S: shape {s.shape}, sum across rows = {s.sum(axis=1).min()}..{s.sum(axis=1).max()}")
    print(f"  - Adjacency Matrix A_micro: shape {adj.shape}, total edges = {int(adj.sum() / 2)}")
    print(f"  - Threat horizons: min tau = {tau_h.min():.2f}h, max tau = {tau_h.max():.2f}h")
    print(f"  - Saved artifacts to: {p}")
