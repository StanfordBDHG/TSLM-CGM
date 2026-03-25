#
# This source file is part of the TSLM-CGM project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

import json
import os
import io
from typing import Tuple, List, Optional
from pathlib import Path
from azure.storage.blob import BlobServiceClient, ContainerClient
from dotenv import load_dotenv
import numpy as np
import pandas as pd

# Azure config
load_dotenv()
ACCOUNT_NAME    = os.environ["AZURE_ACCOUNT_NAME"]
SAS_TOKEN       = os.environ["AZURE_SAS_TOKEN"]
CONTAINER_NAME  = os.environ["AZURE_CONTAINER_NAME"]
DATASET_PREFIX  = os.environ["AZURE_DATASET_PREFIX"]

ACCOUNT_URL = f"https://{ACCOUNT_NAME}.blob.core.windows.net?{SAS_TOKEN}"

CGM_PREFIX = (f"{DATASET_PREFIX}/wearable_blood_glucose/"
              "continuous_glucose_monitoring/dexcom_g6")

# Default local cache directory.
# Save each patient's CGM data as a single parquet file:
#   <CACHE_DIR>/cgm/<patient_id>.parquet
DEFAULT_CACHE_DIR = Path(os.environ.get("TSLM_CACHE_DIR", "./data/cache")).resolve()

# Dexcom G6 reports 'Low' when glucose < 40 mg/dL and 'High' when > 400 mg/dL.
_SENTINEL_VALUES = {"low": 39.0, "high": 401.0}
 
 
def _parse_glucose_value(value) -> float:
    """
    Convert a raw Dexcom glucose value to float.
 
    Handles the string sentinels 'LOW' and 'HIGH' that the Dexcom G6 reports
    when glucose falls outside its measurable range (<40 or >400 mg/dL).
    """
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in _SENTINEL_VALUES:
            return _SENTINEL_VALUES[normalised]
        return float(value)  
    return float(value)


def get_container_client() -> ContainerClient:
    service = BlobServiceClient(account_url=ACCOUNT_URL)
    return service.get_container_client(CONTAINER_NAME)


def _cgm_cache_path(patient_id: str, cache_dir: Path) -> Path:
    return cache_dir / "cgm" / f"{patient_id}.parquet"


def _download_cgm_from_azure(patient_id: str) -> pd.DataFrame:
    """
    Download and parse the raw CGM JSON for one patient from Azure.
    Returns a DataFrame with columns [timestamp, glucose], sorted ascending.
    """
    client = get_container_client()
    blob_path = f"{CGM_PREFIX}/{patient_id}/{patient_id}_DEX.json"
 
    raw = client.get_blob_client(blob_path).download_blob().readall()
    data = json.loads(raw)
 
    rows = []
    for reading in data["body"]["cgm"]:
        if reading.get("event_type") != "EGV":
            continue
        value = reading.get("blood_glucose", {}).get("value")
        if value is None:
            continue
        ts = reading["effective_time_frame"]["time_interval"]["start_date_time"]
        rows.append({"timestamp": ts, "glucose": _parse_glucose_value(value)})
 
    if not rows:
        raise ValueError(f"No valid EGV readings found for patient {patient_id}")
 
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_cgm_for_patient(
    patient_id: str,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Return CGM data for a single patient as a DataFrame.
 
    1. If a local parquet cache exists (and force_download=False), load from disk.
    2. Otherwise, download from Azure, save to disk, then return.
 
    The cache file is a parquet with columns:
        timestamp   datetime64[utc]
        glucose     float64  (mg/dL; LOW→39, HIGH→401)
 
    Parameters:
    patient_id : str
    cache_dir  : Path, optional
        Root cache directory. Defaults to DEFAULT_CACHE_DIR
    force_download : bool
        If True, always re-download from Azure and overwrite the cache.
 
    Returns:
    pd.DataFrame sorted by timestamp ascending.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
 
    cache_path = _cgm_cache_path(patient_id, cache_dir)
 
    if cache_path.exists() and not force_download:
        # Fast path: read from local disk
        df = pd.read_parquet(cache_path)
        # Ensure timezone is preserved correctly after round-trip
        if not hasattr(df["timestamp"].dtype, "tz") or df["timestamp"].dtype.tz is None:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
 
    # Slow path: download from Azure and cache
    print(f"  [cache miss] Downloading CGM for patient {patient_id} from Azure...")
    df = _download_cgm_from_azure(patient_id)
 
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"  [cached] Saved to {cache_path}")
 
    return df
 
 
def prefetch_all_cgm(
    patient_ids: list,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> None:
    """
    Download and cache CGM data for all given patients upfront.
 
    Call this once before starting training to avoid per-epoch Azure latency.
 
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
 
    already_cached = sum(
        1 for pid in patient_ids
        if _cgm_cache_path(pid, cache_dir).exists()
    )
    to_download = len(patient_ids) - already_cached
    print(f"[prefetch] {already_cached} patients already cached, "
          f"{to_download} to download.")
 
    failed = []
    for i, pid in enumerate(patient_ids):
        try:
            load_cgm_for_patient(pid, cache_dir=cache_dir, force_download=force_download)
            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{len(patient_ids)}] done")
        except Exception as e:
            failed.append(pid)
            print(f"  [error] {pid}: {e}")
 
    print(f"[prefetch] Complete. {len(failed)} failures: {failed[:5]}"
          f"{'...' if len(failed) > 5 else ''}")
 

def get_cgm_stats(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for a CGM df.
    To use when building the time series text description for model prompt.
    """
    glucose = df["glucose"]

    return {
    "n_readings":    len(glucose),
    "mean":          round(float(glucose.mean()), 1),
    "std":           round(float(glucose.std()), 1),
    "min":           round(float(glucose.min()), 1),
    "max":           round(float(glucose.max()), 1),
    "pct_in_range":  round(float(glucose.between(70, 180).mean() * 100), 1),
    "pct_low":       round(float((glucose < 70).mean() * 100), 1),
    "pct_high":      round(float((glucose > 180).mean() * 100), 1),
}