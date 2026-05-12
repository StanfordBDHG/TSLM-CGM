#
# This source file is part of the TSLM-CGM project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Compute HOMA-IR insulin resistance scores from Azure Blob Storage data.

Steps:
  1. Filter patients by fasting duration >= 8 h  (observation.csv)
  2. Extract BMI, fasting glucose, fasting insulin, WHR   (measurement.csv)
  3. Compute HOMA-IR score and binary IR label
  4. Join train/val/test split labels from participants.tsv
  5. Save patient-level CSV to data/ir_scores.csv
"""

import io
import os
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

from cgm_diabetes.data.cgm_loader import get_container_client

# ── Blob paths (relative to AZURE_DATASET_PREFIX) ────────────────────────────
_MEASUREMENT_BLOB  = "clinical_data/measurement.csv"
_OBSERVATION_BLOB  = "clinical_data/observation.csv"
_PARTICIPANTS_BLOB = "participants.tsv"

# ── Observation filter ────────────────────────────────────────────────────────
_FASTING_OBS_VALUE = "paate, How many hours since you last ate? (number"
_FASTING_MIN_HOURS = 8.0

# ── Measurement source value prefixes (matched with str.startswith) ───────────
_BMI_SRC       = "bmi_vsorres, BMI"
_GLUCOSE_SRC   = "import_glucose, Glucose [Mass/volume] in Serum or"
_INSULIN_SRC   = "import_insulin, Insulin [Units/volume] in Serum o"
_WHR_SRC       = "whr_vsorres, Waist to Hip Ratio (WHR)"

# ── HOMA-IR threshold ─────────────────────────────────────────────────────────
_IR_THRESHOLD = 2.9

# ── Output ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent / "data"
OUTPUT_CSV  = DATA_DIR / "ir_scores.csv"


def _dataset_prefix() -> str:
    load_dotenv()
    return os.environ["AZURE_DATASET_PREFIX"]


def _download_csv(blob_name: str) -> pd.DataFrame:
    """Download a CSV blob into a DataFrame."""
    prefix = _dataset_prefix()
    client = get_container_client()
    blob_path = f"{prefix}/{blob_name}"
    print(f"[azure] Downloading {blob_path} ...")
    raw = client.get_blob_client(blob_path).download_blob().readall()
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    print(f"[azure] {blob_name}: {len(df):,} rows, {df.shape[1]} columns")
    return df


def _filter_fasting_patients(obs_df: pd.DataFrame) -> set:
    """
    Return the set of person_ids who have at least one fasting observation
    with value_as_number >= _FASTING_MIN_HOURS.
    """
    fasting = obs_df[
        obs_df["observation_source_value"] == _FASTING_OBS_VALUE
    ].copy()

    fasting["value_as_number"] = pd.to_numeric(fasting["value_as_number"], errors="coerce")
    fasting = fasting.dropna(subset=["value_as_number"])

    qualifying = fasting[fasting["value_as_number"] >= _FASTING_MIN_HOURS]
    return set(qualifying["person_id"].unique())


def _pivot_measurements(meas_df: pd.DataFrame, valid_ids: set) -> pd.DataFrame:
    """
    Restrict to valid patients, pivot the four measurements to wide format.
    Returns a DataFrame with one row per person_id.
    """
    # Prefix → feature name; startswith matching handles potentially truncated strings
    src_prefixes = [
        (_BMI_SRC,     "bmi"),
        (_GLUCOSE_SRC, "fasting_glucose"),
        (_INSULIN_SRC, "fasting_insulin"),
        (_WHR_SRC,     "waist_to_hip_ratio"),
    ]

    meas_df = meas_df[meas_df["person_id"].isin(valid_ids)].copy()
    meas_df["value_as_number"] = pd.to_numeric(meas_df["value_as_number"], errors="coerce")

    def _map_feature(val: str) -> str | None:
        for prefix, name in src_prefixes:
            if isinstance(val, str) and val.startswith(prefix):
                return name
        return None

    meas_df["feature"] = meas_df["measurement_source_value"].map(_map_feature)
    meas_df = meas_df[meas_df["feature"].notna()].copy()

    # One value per (patient, feature) — take the first non-null if duplicates exist
    pivoted = (
        meas_df
        .dropna(subset=["value_as_number"])
        .sort_values("person_id")
        .groupby(["person_id", "feature"])["value_as_number"]
        .first()
        .unstack("feature")
        .reset_index()
    )

    return pivoted


def _load_split_map() -> dict:
    """Return {person_id: recommended_split} for all participants from participants.tsv."""
    prefix = _dataset_prefix()
    client = get_container_client()
    raw = client.get_blob_client(f"{prefix}/{_PARTICIPANTS_BLOB}").download_blob().readall()
    df = pd.read_csv(io.BytesIO(raw), sep="\t", dtype={"person_id": str})
    return dict(zip(df["person_id"].astype(str), df["recommended_split"]))


def compute_ir_scores() -> pd.DataFrame:
    # ── Step 1: fasting filter ────────────────────────────────────────────────
    obs_df = _download_csv(_OBSERVATION_BLOB)
    all_obs_patients = set(obs_df["person_id"].unique())
    fasted_ids = _filter_fasting_patients(obs_df)
    removed_fasting = len(all_obs_patients) - len(fasted_ids)
    print(
        f"\n[fasting filter] {len(fasted_ids)} patients kept "
        f"(>= {_FASTING_MIN_HOURS:.0f} h fasted); "
        f"{removed_fasting} removed (did not meet threshold or missing observation)"
    )

    # ── Step 2: measurements ──────────────────────────────────────────────────
    meas_df = _download_csv(_MEASUREMENT_BLOB)
    all_meas_patients = set(meas_df[meas_df["person_id"].isin(fasted_ids)]["person_id"].unique())
    wide = _pivot_measurements(meas_df, fasted_ids)

    required_cols = ["bmi", "fasting_glucose", "fasting_insulin", "waist_to_hip_ratio"]
    before_drop = len(wide)
    wide = wide.dropna(subset=required_cols)
    dropped_missing = before_drop - len(wide)
    print(
        f"[measurement pivot] {len(wide)} patients have all four measurements; "
        f"{dropped_missing} dropped (one or more measurements missing)"
    )

    # ── Step 3: HOMA-IR ───────────────────────────────────────────────────────
    # Multiply by 24 to convert ng/mL → uIU/mL before applying standard HOMA-IR
    wide["insulin_resistance_score"] = (
        wide["fasting_insulin"] * 24.0 * wide["fasting_glucose"]
    ) / 405.0

    wide["insulin_resistance"] = wide["insulin_resistance_score"].apply(
        lambda s: "IR" if s >= _IR_THRESHOLD else "Non_IR"
    )

    # ── Step 4: split labels ──────────────────────────────────────────────────
    print("\n[azure] Downloading participants.tsv for split labels ...")
    split_map = _load_split_map()
    wide["split"] = wide["person_id"].astype(str).map(split_map)
    split_counts = wide["split"].value_counts(dropna=False).to_dict()
    print(f"[splits] Distribution: {split_counts}")

    # ── Step 5: output ────────────────────────────────────────────────────────
    output_cols = [
        "person_id",
        "bmi",
        "waist_to_hip_ratio",
        "fasting_glucose",
        "fasting_insulin",
        "insulin_resistance_score",
        "insulin_resistance",
        "split",
    ]
    result = wide[output_cols].reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[output] Saved {len(result)} rows to {OUTPUT_CSV}")

    # Summary
    ir_count    = (result["insulin_resistance"] == "IR").sum()
    non_ir_count = (result["insulin_resistance"] == "Non_IR").sum()
    score_stats  = result["insulin_resistance_score"].describe()
    print(
        f"\n── Summary ──────────────────────────────────────────\n"
        f"  Total patients : {len(result)}\n"
        f"  IR             : {ir_count}\n"
        f"  Non-IR         : {non_ir_count}\n"
        f"  Score stats    :\n{score_stats.to_string()}\n"
        f"─────────────────────────────────────────────────────"
    )

    return result


if __name__ == "__main__":
    compute_ir_scores()
