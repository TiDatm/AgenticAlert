"""
Geographic Adjacency Graph and Spatial Mappings for Ukraine's 25 Administrative Regions
(24 Oblasts + Kyiv City).
"""

import sys
import json
import numpy as np
from typing import Dict, List, Tuple

# Ensure UTF-8 stdout handling on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Canonical list of 25 regions (Oblasts + Kyiv City)
REGIONS: List[str] = [
    "Вінницька область",
    "Волинська область",
    "Дніпропетровська область",
    "Донецька область",
    "Житомирська область",
    "Закарпатська область",
    "Запорізька область",
    "Івано-Франківська область",
    "Київська область",
    "м. Київ",
    "Кіровоградська область",
    "Луганська область",
    "Львівська область",
    "Миколаївська область",
    "Одеська область",
    "Полтавська область",
    "Рівненська область",
    "Сумська область",
    "Тернопільська область",
    "Харківська область",
    "Херсонська область",
    "Хмельницька область",
    "Черкаська область",
    "Чернівецька область",
    "Чернігівська область",
]

REGION_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(REGIONS)}
IDX_TO_REGION: Dict[int, str] = {i: name for i, name in enumerate(REGIONS)}

# Regional groupings by operational threat posture
BORDER_REGIONS: List[str] = [
    "Харківська область",
    "Сумська область",
    "Чернігівська область",
    "Луганська область",
    "Донецька область",
    "Запорізька область",
]

FRONTLINE_REGIONS: List[str] = [
    "Донецька область",
    "Луганська область",
    "Запорізька область",
    "Херсонська область",
    "Харківська область",
]

# Real geographic land border adjacency
ADJACENCY_MAP: Dict[str, List[str]] = {
    "Вінницька область": [
        "Житомирська область", "Київська область", "Черкаська область",
        "Кіровоградська область", "Одеська область", "Хмельницька область", "Чернівецька область"
    ],
    "Волинська область": [
        "Рівненська область", "Львівська область"
    ],
    "Дніпропетровська область": [
        "Полтавська область", "Харківська область", "Донецька область",
        "Запорізька область", "Херсонська область", "Миколаївська область", "Кіровоградська область"
    ],
    "Донецька область": [
        "Луганська область", "Харківська область", "Дніпропетровська область", "Запорізька область"
    ],
    "Житомирська область": [
        "Київська область", "Вінницька область", "Хмельницька область", "Рівненська область"
    ],
    "Закарпатська область": [
        "Львівська область", "Івано-Франківська область"
    ],
    "Запорізька область": [
        "Дніпропетровська область", "Донецька область", "Херсонська область"
    ],
    "Івано-Франківська область": [
        "Львівська область", "Закарпатська область", "Тернопільська область", "Чернівецька область"
    ],
    "Київська область": [
        "м. Київ", "Чернігівська область", "Полтавська область",
        "Черкаська область", "Вінницька область", "Житомирська область"
    ],
    "м. Київ": [
        "Київська область"
    ],
    "Кіровоградська область": [
        "Черкаська область", "Полтавська область", "Дніпропетровська область",
        "Миколаївська область", "Одеська область", "Вінницька область"
    ],
    "Луганська область": [
        "Харківська область", "Донецька область"
    ],
    "Львівська область": [
        "Волинська область", "Рівненська область", "Тернопільська область",
        "Івано-Франківська область", "Закарпатська область"
    ],
    "Миколаївська область": [
        "Кіровоградська область", "Дніпропетровська область", "Херсонська область", "Одеська область"
    ],
    "Одеська область": [
        "Вінницька область", "Кіровоградська область", "Миколаївська область"
    ],
    "Полтавська область": [
        "Чернігівська область", "Сумська область", "Харківська область",
        "Дніпропетровська область", "Кіровоградська область", "Черкаська область", "Київська область"
    ],
    "Рівненська область": [
        "Житомирська область", "Хмельницька область", "Тернопільська область", "Львівська область", "Волинська область"
    ],
    "Сумська область": [
        "Чернігівська область", "Полтавська область", "Харківська область"
    ],
    "Тернопільська область": [
        "Рівненська область", "Хмельницька область", "Чернівецька область",
        "Івано-Франківська область", "Львівська область"
    ],
    "Харківська область": [
        "Сумська область", "Полтавська область", "Дніпропетровська область",
        "Донецька область", "Луганська область"
    ],
    "Херсонська область": [
        "Миколаївська область", "Дніпропетровська область", "Запорізька область"
    ],
    "Хмельницька область": [
        "Рівненська область", "Житомирська область", "Вінницька область",
        "Чернівецька область", "Тернопільська область"
    ],
    "Черкаська область": [
        "Київська область", "Полтавська область", "Кіровоградська область", "Вінницька область"
    ],
    "Чернівецька область": [
        "Івано-Франківська область", "Тернопільська область", "Хмельницька область", "Вінницька область"
    ],
    "Чернігівська область": [
        "Сумська область", "Полтавська область", "Київська область"
    ],
}


def build_adjacency_matrix() -> np.ndarray:
    """
    Constructs a symmetric binary adjacency matrix A of shape (25, 25).
    """
    n = len(REGIONS)
    adj = np.zeros((n, n), dtype=np.float32)

    for region, neighbors in ADJACENCY_MAP.items():
        if region not in REGION_TO_IDX:
            continue
        u = REGION_TO_IDX[region]
        for neighbor in neighbors:
            if neighbor in REGION_TO_IDX:
                v = REGION_TO_IDX[neighbor]
                adj[u, v] = 1.0
                adj[v, u] = 1.0

    return adj


def get_normalized_laplacian(adj: np.ndarray) -> np.ndarray:
    """
    Calculates symmetric normalized Laplacian with self-loops:
    A_norm = D^(-1/2) * (A + I) * D^(-1/2)
    """
    adj_loop = adj + np.eye(adj.shape[0], dtype=np.float32)
    degrees = np.sum(adj_loop, axis=1)
    d_inv_sqrt = np.zeros_like(degrees, dtype=np.float32)
    positive_mask = degrees > 0
    d_inv_sqrt[positive_mask] = np.power(degrees[positive_mask], -0.5)
    d_mat = np.diag(d_inv_sqrt)
    return np.matmul(np.matmul(d_mat, adj_loop), d_mat).astype(np.float32)


# Canonical region geographic centroids (latitude, longitude)
REGION_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "Вінницька область": (49.23, 28.47),
    "Волинська область": (50.75, 25.33),
    "Дніпропетровська область": (48.46, 35.04),
    "Донецька область": (48.01, 37.80),
    "Житомирська область": (50.25, 28.66),
    "Закарпатська область": (48.62, 22.29),
    "Запорізька область": (47.83, 35.14),
    "Івано-Франківська область": (48.92, 24.71),
    "Київська область": (50.35, 30.65),
    "м. Київ": (50.45, 30.52),
    "Кіровоградська область": (48.50, 32.26),
    "Луганська область": (48.57, 39.30),
    "Львівська область": (49.84, 24.03),
    "Миколаївська область": (46.97, 31.99),
    "Одеська область": (46.48, 30.72),
    "Полтавська область": (49.58, 34.55),
    "Рівненська область": (50.62, 26.25),
    "Сумська область": (50.90, 34.80),
    "Тернопільська область": (49.55, 25.59),
    "Харківська область": (49.99, 36.23),
    "Херсонська область": (46.63, 32.61),
    "Хмельницька область": (49.42, 26.98),
    "Черкаська область": (49.44, 32.06),
    "Чернівецька область": (48.29, 25.93),
    "Чернігівська область": (51.50, 31.30),
}


def build_directed_transition_matrices() -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs dual directed transition matrices for advective threat flow:
    - P_fwd = D_out^(-1) * A_dir (Downstream threat flow from staging/border inwards to west)
    - P_bwd = D_in^(-1) * A_dir.T (Upstream origin tracking)
    """
    n = len(REGIONS)
    adj_sym = build_adjacency_matrix()
    adj_dir = np.zeros((n, n), dtype=np.float32)

    # Threats physically enter from East/North/South and propagate West/North-West
    # Assign edge direction u -> v if centroid_lon(u) > centroid_lon(v) or threat ingress
    for i in range(n):
        reg_i = REGIONS[i]
        c_i = REGION_CENTROIDS.get(reg_i, (50.0, 30.0))
        for j in range(n):
            if adj_sym[i, j] > 0:
                reg_j = REGIONS[j]
                c_j = REGION_CENTROIDS.get(reg_j, (50.0, 30.0))
                
                # Forward advection weight: East to West flow (lon_i >= lon_j - 0.5)
                # or North/South border ingress into interior
                is_fwd = (c_i[1] >= c_j[1] - 0.2) or (reg_i in ["Крим", "Херсонська область", "Сумська область", "Чернігівська область"])
                if is_fwd:
                    adj_dir[i, j] = 1.0
                else:
                    adj_dir[i, j] = 0.35  # Minor bidirectional permeability

    # Add self loops
    adj_fwd = adj_dir + np.eye(n, dtype=np.float32)
    d_out = np.maximum(np.sum(adj_fwd, axis=1, keepdims=True), 1.0)
    p_fwd = (adj_fwd / d_out).astype(np.float32)

    adj_bwd = adj_dir.T + np.eye(n, dtype=np.float32)
    d_in = np.maximum(np.sum(adj_bwd, axis=1, keepdims=True), 1.0)
    p_bwd = (adj_bwd / d_in).astype(np.float32)

    return p_fwd, p_bwd


def get_static_distance_features() -> np.ndarray:
    """
    Returns static normalized distance metrics [25, 3] for each region:
    - Dim 0: Normalized distance to Russian Federation border
    - Dim 1: Normalized distance to active frontline
    - Dim 2: Normalized distance to Black Sea / Crimean staging envelope
    """
    n = len(REGIONS)
    feats = np.zeros((n, 3), dtype=np.float32)

    # Reference coordinates
    rf_border = np.array([51.5, 35.5])       # East/NE border envelope
    frontline = np.array([48.0, 37.0])       # Donbas/Zaporizhzhia contact line
    black_sea = np.array([45.0, 33.5])       # Crimea / Black Sea maritime axis

    for i, reg_name in enumerate(REGIONS):
        c = np.array(REGION_CENTROIDS.get(reg_name, (50.0, 30.0)))
        d_rf = np.linalg.norm(c - rf_border) * 111.0  # Approx km
        d_fl = np.linalg.norm(c - frontline) * 111.0
        d_bs = np.linalg.norm(c - black_sea) * 111.0

        feats[i, 0] = np.clip(d_rf / 1200.0, 0.0, 1.0)
        feats[i, 1] = np.clip(d_fl / 1200.0, 0.0, 1.0)
        feats[i, 2] = np.clip(d_bs / 1000.0, 0.0, 1.0)

    return feats


# Canonical Kinematic Isochrone Horizons according to flight kinematics blueprint
REGIONAL_ISOCHRONES: Dict[str, float] = {
    # Border & Tactical Frontline (1.0h / 4 steps)
    "Харківська область": 1.0,
    "Сумська область": 1.0,
    "Чернігівська область": 1.0,
    "Донецька область": 1.0,
    "Луганська область": 1.0,
    "Запорізька область": 1.0,
    "Дніпропетровська область": 1.0,
    "Херсонська область": 1.0,
    # Central & Southern Transit Corridors (2.0h / 8 steps)
    "м. Київ": 2.0,
    "Київська область": 2.0,
    "Полтавська область": 2.0,
    "Миколаївська область": 2.0,
    "Одеська область": 2.0,
    "Черкаська область": 2.0,
    "Кіровоградська область": 2.0,
    # West-Central Incursion Axis (3.0h / 12 steps)
    "Вінницька область": 3.0,
    "Житомирська область": 3.0,
    "Хмельницька область": 3.0,
    "Чернівецька область": 3.0,
    # Deep Rear Defensive Buffer (4.5h / 18 steps)
    "Львівська область": 4.5,
    "Рівненська область": 4.5,
    "Тернопільська область": 4.5,
    "Волинська область": 4.5,
    "Івано-Франківська область": 4.5,
    "Закарпатська область": 4.5,
}

REGIONAL_ISOCHRONE_STEPS: Dict[str, int] = {
    k: int(round(v * 4)) for k, v in REGIONAL_ISOCHRONES.items()
}


def compute_regional_threat_distances() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes for each of the 25 administrative regions:
    1. Threat distance d_threat(r) = min(d_border, d_frontline, d_coast) in km.
    2. Kinematic isochrone warning horizon tau_isochrone(r) in hours from {1.0, 2.0, 3.0, 4.5}.
    3. Isochrone steps (in 15m steps) from {4, 8, 12, 18}.
    """
    rf_border = np.array([51.5, 35.5])       # East/NE border envelope
    frontline = np.array([48.0, 37.0])       # Donbas contact line
    black_sea = np.array([45.0, 33.5])       # Crimea / Black Sea maritime axis

    n = len(REGIONS)
    d_threats = np.zeros(n, dtype=np.float32)
    tau_hours = np.zeros(n, dtype=np.float32)
    tau_steps = np.zeros(n, dtype=np.int32)

    for i, reg_name in enumerate(REGIONS):
        c = np.array(REGION_CENTROIDS.get(reg_name, (50.0, 30.0)))
        d_rf = float(np.linalg.norm(c - rf_border) * 111.0)
        d_fl = float(np.linalg.norm(c - frontline) * 111.0)
        d_bs = float(np.linalg.norm(c - black_sea) * 111.0)
        d_min = min(d_rf, d_fl, d_bs)
        d_threats[i] = d_min

        # Lookup calibrated kinematic isochrone
        tau = REGIONAL_ISOCHRONES.get(reg_name, 2.0)
        steps = REGIONAL_ISOCHRONE_STEPS.get(reg_name, int(round(tau * 4)))
        tau_hours[i] = tau
        tau_steps[i] = steps

    return d_threats, tau_hours, tau_steps


if __name__ == "__main__":
    adj = build_adjacency_matrix()
    lap = get_normalized_laplacian(adj)
    p_fwd, p_bwd = build_directed_transition_matrices()
    d_feats = get_static_distance_features()
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Directed P_fwd shape: {p_fwd.shape}, P_bwd shape: {p_bwd.shape}")
    print(f"Distance features shape: {d_feats.shape}")
    print(f"Sample distance features (Kyiv): {d_feats[REGIONS.index('м. Київ')]}")

