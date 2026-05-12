#
# This source file is part of the TSLM-CGM project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from cgm_diabetes.data.cgm_loader import load_cgm_for_patient, DEFAULT_CACHE_DIR
from cgm_diabetes.ir_prediction.cgm_features import extract_cgm_features

DATA_DIR = Path(__file__).parent / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV  = DATA_DIR / "test.csv"

# Column that identifies the patient in the CSVs
_ID_COL    = "participant_id"
_LABEL_COL = "IR_label"


_LABEL_MAP = {"IR": 1, "Non.IR": 0, "non.ir": 0, "ir": 1}


def _encode_label(series: pd.Series) -> pd.Series:
    """Convert string IR labels to 0/1 integers; leave numeric labels unchanged."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    encoded = series.map(_LABEL_MAP)
    unknown = series[encoded.isna()].unique()
    if len(unknown):
        raise ValueError(f"Unknown IR_label values: {unknown}. Expected 'IR' or 'Non.IR'.")
    return encoded.astype(int)


def _load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if _ID_COL not in df.columns:
        raise ValueError(f"Expected column '{_ID_COL}' in {path}")
    df[_ID_COL] = df[_ID_COL].astype(str)
    if _LABEL_COL in df.columns:
        df[_LABEL_COL] = _encode_label(df[_LABEL_COL])
    return df


def _build_cgm_feature_df(
    patient_ids: List[str],
    cache_dir: Path,
    skip_on_error: bool = True,
) -> pd.DataFrame:
    """Download (or load from cache) each patient's CGM trace and compute features."""
    rows = []
    for i, pid in enumerate(patient_ids):
        try:
            cgm_df = load_cgm_for_patient(pid, cache_dir=cache_dir)
            feats  = extract_cgm_features(cgm_df)
            feats[_ID_COL] = pid
            rows.append(feats)
        except Exception as e:
            if skip_on_error:
                print(f"  [warning] CGM features skipped for {pid}: {e}")
            else:
                raise
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(patient_ids)}] CGM features computed")

    return pd.DataFrame(rows)


def load_split(
    split: str = "train",
    include_cgm_features: bool = True,
    cache_dir: Optional[Path] = None,
    skip_on_error: bool = True,
) -> pd.DataFrame:
    """
    Load one split (train or test) with optional CGM-derived features merged in.

    Parameters
    ----------
    split : "train" or "test"
    include_cgm_features : whether to fetch full CGM traces and compute features
    cache_dir : CGM parquet cache root (defaults to cgm_loader.DEFAULT_CACHE_DIR)
    skip_on_error : if True, patients whose CGM download fails are kept in the
                    dataset with NaN CGM features instead of raising

    Returns
    -------
    DataFrame with original CSV columns plus any cgm_* feature columns.
    Rows are indexed by participant_id (as a column, not the index).
    """
    if split == "train":
        path = TRAIN_CSV
    elif split == "test":
        path = TEST_CSV
    else:
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found: {path}\n"
            f"Please place your {split}.csv file in {DATA_DIR}/"
        )

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    df = _load_raw(path)
    print(f"[data_loader] Loaded {len(df)} rows from {path.name}")

    if not include_cgm_features:
        return df

    print(f"[data_loader] Computing CGM features for {len(df)} patients...")
    cgm_feat_df = _build_cgm_feature_df(
        df[_ID_COL].tolist(),
        cache_dir=cache_dir,
        skip_on_error=skip_on_error,
    )

    if cgm_feat_df.empty or _ID_COL not in cgm_feat_df.columns:
        print("[data_loader] No CGM features were computed — returning base CSV only.")
        return df

    merged = df.merge(cgm_feat_df, on=_ID_COL, how="left")
    n_cgm  = cgm_feat_df[_ID_COL].nunique()
    print(f"[data_loader] Merged CGM features for {n_cgm}/{len(df)} patients.")

    # ── Interaction features (CGM × anthropometric, CGM × CGM) ──────────────
    _BMI_COL = "bmi_vsorres..BMI"
    _WHR_COL = "whr_vsorres..Waist.to.Hip.Ratio..WHR."

    # Existing: hyper duration / BMI, postprandial AUC / WHR
    if "cgm_hyper_episode_duration_mean" in merged.columns and _BMI_COL in merged.columns:
        merged["cgm_hyper_duration_per_bmi"] = (
            merged["cgm_hyper_episode_duration_mean"] / merged[_BMI_COL].replace(0, np.nan)
        )
    if "cgm_postprandial_auc_2h" in merged.columns and _WHR_COL in merged.columns:
        merged["cgm_postprandial_auc_per_whr"] = (
            merged["cgm_postprandial_auc_2h"] / merged[_WHR_COL].replace(0, np.nan)
        )

    # Total hyperglycemic burden / BMI (burden normalised for adiposity)
    if "cgm_auc_above_180" in merged.columns and _BMI_COL in merged.columns:
        merged["cgm_auc_above_180_per_bmi"] = (
            merged["cgm_auc_above_180"] / merged[_BMI_COL].replace(0, np.nan)
        )

    # Reactive hypos × central adiposity (WHR amplifies compensatory hyperinsulinaemia signal)
    if "cgm_n_hypo_l2_events" in merged.columns and _WHR_COL in merged.columns:
        merged["cgm_n_hypo_l2_x_whr"] = (
            merged["cgm_n_hypo_l2_events"] * merged[_WHR_COL]
        )

    # Slow clearance × total hyperglycemic burden (product of two top CGM features)
    if ("cgm_hyper_episode_duration_mean" in merged.columns
            and "cgm_auc_above_180" in merged.columns):
        merged["cgm_hyper_duration_x_auc"] = (
            merged["cgm_hyper_episode_duration_mean"] * merged["cgm_auc_above_180"]
        )

    # Slow clearance × reactive hypos (two independent IR mechanisms, joint signal)
    if ("cgm_hyper_episode_duration_mean" in merged.columns
            and "cgm_n_hypo_l2_events" in merged.columns):
        merged["cgm_hyper_duration_x_hypo"] = (
            merged["cgm_hyper_episode_duration_mean"] * merged["cgm_n_hypo_l2_events"]
        )

    return merged


def get_feature_columns(df: pd.DataFrame, feature_set: str = "all") -> List[str]:
    """
    Return a list of feature column names for a given feature set name.

    feature_set options
    -------------------
    "base"      – original CSV columns only (no CGM-derived, no id/label)
    "cgm"       – only the cgm_* columns extracted from raw traces
    "all"       – base + cgm combined
    """
    # Exclude id, label, and any unnamed index columns the CSV may carry
    exclude = {_ID_COL, _LABEL_COL}
    all_cols  = [c for c in df.columns
                 if c not in exclude and not c.startswith("Unnamed:")]
    base_cols = [c for c in all_cols if not c.startswith("cgm_")]
    cgm_cols  = [c for c in all_cols if c.startswith("cgm_")]

    if feature_set == "base":
        return base_cols
    if feature_set == "cgm":
        return cgm_cols
    if feature_set == "all":
        return all_cols
    # Allow a custom list passed as a comma-separated string
    return [c.strip() for c in feature_set.split(",") if c.strip() in df.columns]
