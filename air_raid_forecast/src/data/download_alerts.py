"""
Module for downloading and caching the official air raid alerts dataset.
Source: Vadimkin/ukrainian-air-raid-sirens-dataset (GitHub)
"""

import os
import sys
from pathlib import Path
import requests
import pandas as pd
from typing import Optional

# Ensure project root and parent are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

RAW_DATA_URL = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_uk.csv"
ALT_DATA_URL = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/master/datasets/official_data_uk.csv"


def resolve_data_path(file_path: Optional[str] = None) -> str:
    """
    Resolves data file path cleanly whether run from root, src, or parent directories.
    """
    if file_path and os.path.exists(file_path):
        return os.path.abspath(file_path)

    candidates = [
        file_path,
        "data/raw/official_data_uk.csv",
        "air_raid_forecast/data/raw/official_data_uk.csv",
        str(_PROJECT_ROOT / "data" / "raw" / "official_data_uk.csv"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)

    if file_path:
        return os.path.abspath(file_path)
    return str(_PROJECT_ROOT / "data" / "raw" / "official_data_uk.csv")


def download_official_alerts(
    output_path: Optional[str] = None,
    force_redownload: bool = False
) -> str:
    """
    Downloads official Ukrainian air raid dataset if not already present.
    Returns the absolute path to the downloaded CSV file.
    """
    resolved_path = resolve_data_path(output_path)
    os.makedirs(os.path.dirname(resolved_path), exist_ok=True)

    if os.path.exists(resolved_path) and not force_redownload:
        print(f"Dataset already exists at: {resolved_path}")
        return resolved_path

    print(f"Downloading official air alerts dataset from {RAW_DATA_URL}...")
    headers = {"User-Agent": "AirRaidForecast/1.0"}
    
    response = requests.get(RAW_DATA_URL, headers=headers, timeout=60)
    if response.status_code != 200:
        print(f"Primary URL returned {response.status_code}. Trying alternative master branch URL...")
        response = requests.get(ALT_DATA_URL, headers=headers, timeout=60)
        
    response.raise_for_status()

    with open(resolved_path, "wb") as f:
        f.write(response.content)

    file_size_mb = os.path.getsize(resolved_path) / (1024 * 1024)
    print(f"Download complete: {resolved_path} ({file_size_mb:.2f} MB)")
    return resolved_path


def load_and_clean_alerts(
    file_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Loads raw CSV, standardizes column names, parses UTC datetimes,
    and sorts chronologically.
    """
    resolved_path = resolve_data_path(file_path)
    if not os.path.exists(resolved_path):
        resolved_path = download_official_alerts(resolved_path)

    df = pd.read_csv(resolved_path)
    
    # Standardize column names (strip whitespace and lowercase)
    df.columns = [c.strip().lower() for c in df.columns]

    # Map column names if started_at / finished_at are present
    col_mapping = {
        "started_at": "start",
        "finished_at": "end",
        "oblast": "region_title"
    }
    df = df.rename(columns=col_mapping)

    # Convert start and end timestamps to UTC datetime
    if "start" in df.columns:
        df["start"] = pd.to_datetime(df["start"], utc=True, errors="coerce")
    if "end" in df.columns:
        df["end"] = pd.to_datetime(df["end"], utc=True, errors="coerce")

    # Drop records with invalid start times
    df = df.dropna(subset=["start"]).copy()
    
    # Sort chronologically by start time
    df = df.sort_values(by="start").reset_index(drop=True)
    return df


if __name__ == "__main__":
    path = download_official_alerts()
    df = load_and_clean_alerts(path)
    print(f"Loaded {len(df)} alert records. Date range: {df['start'].min()} to {df['start'].max()}")
    print("Columns:", list(df.columns))
    print("Sample:\n", df.head(3))
