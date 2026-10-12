"""
Deterministic Text & Tactical Intelligence Signal Parser.
Extracts high-priority threat indicators from official Air Force and alert messages
in < 1ms on CPU with zero LLM API dependency.
"""

import re
import sys
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Ensure stdout uses UTF-8 if available
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Canonical list of 6 tactical feature keys
TACTICAL_FEATURE_NAMES: List[str] = [
    "flag_mig31k",
    "flag_ballistics",
    "flag_strat_aviation",
    "flag_shahed_group",
    "flag_kab_tactical",
    "flag_clearance",
]

# 25 Canonical Ukrainian regions
CANONICAL_REGIONS: List[str] = [
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

REGION_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CANONICAL_REGIONS)}

# Compiled regex patterns for the 6 tactical threat indicators
RE_MIG31K = re.compile(
    r'(?<!відбій\s)(зліт|активність|виліт|загроза)\s+.*?(міг-31к|mig-31k|саваслейка|кинджал|кінжал)',
    re.IGNORECASE | re.DOTALL
)

RE_BALLISTICS = re.compile(
    r'(?<!відбій\s)(пуск\w*|загроза\s+застосування|вихід|активність)\s+.*?(балісти[чк]|іскандер|с-300|с-400)',
    re.IGNORECASE | re.DOTALL
)

RE_STRAT_AVIATION = re.compile(
    r'(?<!відбій\s)(зліт|активність|виліт|загроза)\s+.*?(\d+\s*борт|ту-95|ту-22|ту-160|оленья|енгельс|стратегіч)',
    re.IGNORECASE | re.DOTALL
)

RE_SHAHED_GROUP = re.compile(
    r'(?<!відбій\s)(група|рух|чергова\s+група|пуск\w*|активність)\s+.*?(шахед|бпла|дронів|шахід|гербер)',
    re.IGNORECASE | re.DOTALL
)

RE_KAB_TACTICAL = re.compile(
    r'(?<!відбій\s)(пуск\w*|скид\w*|активність|загроза)\s+.*?(каб|фаб|керован\w+\s+авіабомб|су-34|су-35|тактичн\w+\s+авіац)',
    re.IGNORECASE | re.DOTALL
)

RE_CLEARANCE = re.compile(
    r'(відбій\s+(загрози|тривоги|повітряної\s+тривоги|по\s+міг|балістичної\s+загрози)?|\bчисто\b|локаційно\s+втрачено|\bзбито\b|перестала\s+існувати)',
    re.IGNORECASE
)

# Spatial Region Keywords Mapping
REGION_KEYWORD_PATTERNS: Dict[str, re.Pattern] = {
    "Вінницька область": re.compile(r'вінниц|вінничч', re.IGNORECASE),
    "Волинська область": re.compile(r'волин|луцьк', re.IGNORECASE),
    "Дніпропетровська область": re.compile(r'дніпро|дніпропетров', re.IGNORECASE),
    "Донецька область": re.compile(r'донецьк|донечч|краматорськ|слов.янськ', re.IGNORECASE),
    "Житомирська область": re.compile(r'житомир', re.IGNORECASE),
    "Закарпатська область": re.compile(r'закарпат|ужгород', re.IGNORECASE),
    "Запорізька область": re.compile(r'запоріз|запоріж', re.IGNORECASE),
    "Івано-Франківська область": re.compile(r'івано-франків|прикарпат', re.IGNORECASE),
    "Київська область": re.compile(r'київщин|київськ\w+\s+обл', re.IGNORECASE),
    "м. Київ": re.compile(r'\bкиїв\b|м\.\s*київ', re.IGNORECASE),
    "Кіровоградська область": re.compile(r'кіровоград|кропивницьк', re.IGNORECASE),
    "Луганська область": re.compile(r'луган', re.IGNORECASE),
    "Львівська область": re.compile(r'львів', re.IGNORECASE),
    "Миколаївська область": re.compile(r'миколаїв', re.IGNORECASE),
    "Одеська область": re.compile(r'одес', re.IGNORECASE),
    "Полтавська область": re.compile(r'полтав|кременчук', re.IGNORECASE),
    "Рівненська область": re.compile(r'рівнен|рівн\b', re.IGNORECASE),
    "Сумська область": re.compile(r'сум\b|сумськ|сумщин', re.IGNORECASE),
    "Тернопільська область": re.compile(r'тернопіль|тернопільщ', re.IGNORECASE),
    "Харківська область": re.compile(r'харків|харківщ', re.IGNORECASE),
    "Херсонська область": re.compile(r'херсон', re.IGNORECASE),
    "Хмельницька область": re.compile(r'хмельницьк|старокостянтинів', re.IGNORECASE),
    "Черкаська область": re.compile(r'черкас|умань', re.IGNORECASE),
    "Чернівецька область": re.compile(r'чернівц|чернівечч|буковин', re.IGNORECASE),
    "Чернігівська область": re.compile(r'чернігів|ніжин', re.IGNORECASE),
}

# Corridor Direction Mappings with inflection support
DIRECTION_PATTERNS: Dict[str, Tuple[re.Pattern, List[str]]] = {
    "північний схід": (re.compile(r'північн\w+\s+сход|північно-схід', re.IGNORECASE), ["Сумська область", "Харківська область", "Чернігівська область", "Полтавська область"]),
    "південний схід": (re.compile(r'південн\w+\s+сход|південно-схід', re.IGNORECASE), ["Запорізька область", "Донецька область", "Дніпропетровська область"]),
    "південний захід": (re.compile(r'південн\w+\s+заход|південно-захід', re.IGNORECASE), ["Одеська область", "Вінницька область", "Чернівецька область"]),
    "північний захід": (re.compile(r'північн\w+\s+заход|північно-захід', re.IGNORECASE), ["Волинська область", "Рівненська область", "Житомирська область"]),
    "північ": (re.compile(r'північ|півноч', re.IGNORECASE), ["Чернігівська область", "Сумська область", "Київська область", "м. Київ", "Житомирська область"]),
    "південь": (re.compile(r'півд(ень|ня|ню|нем|ні)', re.IGNORECASE), ["Одеська область", "Миколаївська область", "Херсонська область", "Запорізька область"]),
    "схід": (re.compile(r'сход(у|і|ом)|схід', re.IGNORECASE), ["Харківська область", "Донецька область", "Луганська область", "Дніпропетровська область"]),
    "захід": (re.compile(r'заход(у|і|ом)|захід', re.IGNORECASE), ["Львівська область", "Волинська область", "Рівненська область", "Тернопільська область", "Івано-Франківська область", "Закарпатська область", "Чернівецька область"]),
    "центр": (re.compile(r'центр', re.IGNORECASE), ["Черкаська область", "Кіровоградська область", "Полтавська область", "Київська область", "м. Київ"]),
}

RE_NATIONWIDE = re.compile(
    r'(по\s+всій\s+території|вся\s+україна|вся\s+територія|масштабна\s+тривога)',
    re.IGNORECASE
)


def extract_threat_corridor(text: str) -> Dict[str, Any]:
    """
    Extracts affected spatial corridors and Ukrainian oblast entities from text.
    Returns:
    - regions: list of directly mentioned canonical region names
    - directions: list of detected flight direction corridors
    - region_mask: np.ndarray [25] with 1.0 for affected regions, 0.0 otherwise
    - is_nationwide: bool, whether alert applies to the whole territory
    """
    if not text:
        return {
            "regions": [],
            "directions": [],
            "region_mask": np.zeros(len(CANONICAL_REGIONS), dtype=np.float32),
            "is_nationwide": False,
        }

    is_nationwide = bool(RE_NATIONWIDE.search(text)) or bool(RE_MIG31K.search(text))
    matched_regions = set()
    matched_directions = []

    if is_nationwide:
        return {
            "regions": list(CANONICAL_REGIONS),
            "directions": ["nationwide"],
            "region_mask": np.ones(len(CANONICAL_REGIONS), dtype=np.float32),
            "is_nationwide": True,
        }

    # Match direct oblast mentions
    for reg_name, pattern in REGION_KEYWORD_PATTERNS.items():
        if pattern.search(text):
            matched_regions.add(reg_name)

    # Match cardinal/corridor directions
    for dir_name, (dir_pat, dir_regions) in DIRECTION_PATTERNS.items():
        if dir_pat.search(text):
            matched_directions.append(dir_name)
            for r in dir_regions:
                matched_regions.add(r)

    mask = np.zeros(len(CANONICAL_REGIONS), dtype=np.float32)
    for r in matched_regions:
        if r in REGION_TO_IDX:
            mask[REGION_TO_IDX[r]] = 1.0

    return {
        "regions": sorted(list(matched_regions)),
        "directions": matched_directions,
        "region_mask": mask,
        "is_nationwide": False,
    }


def parse_tactical_text(text: str) -> Dict[str, float]:
    """
    Parses a single official announcement and extracts 6 tactical threat indicators.
    Returns binary indicators [0.0, 1.0].
    """
    if not text:
        return {k: 0.0 for k in TACTICAL_FEATURE_NAMES}

    is_clear = bool(RE_CLEARANCE.search(text))

    return {
        "flag_mig31k": 1.0 if (not is_clear and bool(RE_MIG31K.search(text))) else 0.0,
        "flag_ballistics": 1.0 if (not is_clear and bool(RE_BALLISTICS.search(text))) else 0.0,
        "flag_strat_aviation": 1.0 if (not is_clear and bool(RE_STRAT_AVIATION.search(text))) else 0.0,
        "flag_shahed_group": 1.0 if (not is_clear and bool(RE_SHAHED_GROUP.search(text))) else 0.0,
        "flag_kab_tactical": 1.0 if (not is_clear and bool(RE_KAB_TACTICAL.search(text))) else 0.0,
        "flag_clearance": 1.0 if is_clear else 0.0,
    }


def parse_batch_messages(messages: List[str]) -> Dict[str, float]:
    """
    Aggregates threat indicators across multiple recent messages (e.g. past 15-60 min).
    """
    agg = {k: 0.0 for k in TACTICAL_FEATURE_NAMES}
    for msg in messages:
        res = parse_tactical_text(msg)
        for k in agg:
            agg[k] = max(agg[k], res[k])
    return agg


if __name__ == "__main__":
    test_samples = [
        "Увага! Зафіксовано зліт МіГ-31К з аеродрому Саваслейка. Ракетна небезпека по всій території України!",
        "Загроза застосування балістичного озброєння зі сходу! Не ігноруйте сигнали тривоги!",
        "Група ворожих БпЛА типу Shahed з півдня курсом на Миколаївщину!",
        "Активність ворожої тактичної авіації вздовж кордону на північному сході! Загроза застосування КАБ на Харківщині та Сумщині!",
        "Відбій загрози по МіГ-31К. По областях відбій тривог."
    ]
    for s in test_samples:
        print(f"Text: {s[:55]}...")
        print(f"Flags: {parse_tactical_text(s)}")
        corridor = extract_threat_corridor(s)
        print(f"Corridor: regions={len(corridor['regions'])}, dirs={corridor['directions']}, nationwide={corridor['is_nationwide']}\n")
