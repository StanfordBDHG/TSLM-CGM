#
# This source file is part of the TSLM-CGM project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from cgm_diabetes.data.cgm_loader import (
    DEFAULT_CACHE_DIR,
    get_cgm_stats,
    load_cgm_for_patient,
    prefetch_all_cgm,
)

READINGS_PER_HOUR = 12  # Dexcom G6: one reading every 5 minutes
MIN_DAYS          = 8
MIN_READINGS      = MIN_DAYS * 24 * READINGS_PER_HOUR  # 2304 readings

LABELS = ["IR", "Non_IR"]

IR_SCORES_CSV  = Path(__file__).parent / "data" / "ir_scores.csv"
IR_LABELS_JSON = Path(__file__).parent / "ir_labels.json"


class CGMIRDataset(Dataset):
    """
    PyTorch Dataset for CGM-based insulin resistance classification.

    Each sample is the full CGM recording for a single patient paired with a
    chain-of-thought reasoning label and a binary IR label. Patient-level
    clinical features (BMI, waist-to-hip ratio) are incorporated into the prompt.

    Patient list and IR labels are loaded from ir_scores.csv (produced by
    compute_ir_scores.py). CGM traces are loaded from Azure Blob Storage or
    a local parquet cache.

    Parameters
    ----------
    split : str
        One of "train", "val", "test".
    EOS_TOKEN : str
        End-of-sequence token string from the model tokenizer.
    min_days : float
        Minimum recording length in days; shorter recordings are skipped.
    max_samples : int or None
        Optional cap on total samples, used for smoke-testing.
    labels_path : str or None
        Path to JSON file mapping person_id -> reasoning string.
        If None, a template fallback is used.
        Generate labels with cgm_diabetes/ir_prediction/generate_ir_labels.py.
    cache_dir : Path or None
        Root directory for local CGM parquet cache.
    force_download : bool
        If True, re-download from Azure even if a cache file already exists.
    """

    def __init__(
        self,
        split: str = "train",
        EOS_TOKEN: str = "",
        min_days: float = MIN_DAYS,
        max_samples: Optional[int] = None,
        labels_path: Optional[str] = None,
        time_series_format_function: Optional[Callable] = None,
        format_sample_str: bool = True,
        cache_dir: Optional[Path] = None,
        force_download: bool = False,
    ):
        super().__init__()

        assert split in ("train", "val", "test", "validation"), (
            f"split must be 'train', 'val', 'validation', or 'test', got '{split}'"
        )
        if split == "validation":
            split = "val"

        self.split          = split
        self.eos_token      = EOS_TOKEN
        self.format_fn      = time_series_format_function
        self.cache_dir      = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.force_download = force_download
        self.min_readings   = int(min_days * 24 * READINGS_PER_HOUR)

        self.labels: Dict[str, str] = {}
        if labels_path is not None:
            with open(labels_path) as f:
                self.labels = json.load(f)
            print(f"[CGMIRDataset] Loaded {len(self.labels)} IR labels from {labels_path}")

        print(f"[CGMIRDataset] Building {split} sample index...")
        self._patient_cache: Dict[str, np.ndarray] = {}
        self.samples = self._build_sample_index(split, max_samples)
        print(f"[CGMIRDataset] {len(self.samples)} patients "
              f"(min {min_days} days = {self.min_readings} readings)")

    # Prefetch

    @classmethod
    def prefetch(
        cls,
        split: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        force_download: bool = False,
    ) -> None:
        """Download and cache CGM data for all IR-dataset patients upfront."""
        if not IR_SCORES_CSV.exists():
            raise FileNotFoundError(
                f"IR scores CSV not found: {IR_SCORES_CSV}\n"
                "Run compute_ir_scores.py first."
            )
        df = pd.read_csv(IR_SCORES_CSV, dtype={"person_id": str})
        if split is not None:
            df = df[df["split"] == split]
        prefetch_all_cgm(
            df["person_id"].tolist(),
            cache_dir=cache_dir,
            force_download=force_download,
        )

    # Index building

    def _load_glucose_array(self, patient_id: str) -> np.ndarray:
        df = load_cgm_for_patient(
            patient_id,
            cache_dir=self.cache_dir,
            force_download=self.force_download,
        )
        return df["glucose"].to_numpy(dtype=np.float32)

    def _build_sample_index(
        self, split: str, max_samples: Optional[int]
    ) -> List[Dict]:
        if not IR_SCORES_CSV.exists():
            raise FileNotFoundError(
                f"IR scores CSV not found: {IR_SCORES_CSV}\n"
                "Run compute_ir_scores.py first."
            )

        df = pd.read_csv(IR_SCORES_CSV, dtype={"person_id": str})
        df = df[df["split"] == split].reset_index(drop=True)

        samples = []
        failed  = []
        skipped = []

        for _, row in df.iterrows():
            if max_samples is not None and len(samples) >= max_samples:
                break

            patient_id = str(row["person_id"])
            try:
                glucose = self._load_glucose_array(patient_id)

                if len(glucose) < self.min_readings:
                    skipped.append(patient_id)
                    print(f"  [skip] {patient_id}: only {len(glucose)} readings "
                          f"(need at least {self.min_readings})")
                    continue

                self._patient_cache[patient_id] = glucose

                samples.append({
                    "patient_id": patient_id,
                    "label":      row["insulin_resistance"],
                    "bmi":        float(row["bmi"]),
                    "whr":        float(row["waist_to_hip_ratio"]),
                    "n_readings": len(glucose),
                })

            except Exception as e:
                failed.append(patient_id)
                print(f"  [error] {patient_id}: {e}")

        if skipped:
            print(f"[CGMIRDataset] Skipped {len(skipped)} patients with fewer than "
                  f"{self.min_readings / READINGS_PER_HOUR / 24:.1f} days of data")
        if failed:
            print(f"[CGMIRDataset] Failed to load {len(failed)} patients: "
                  f"{failed[:5]}{'...' if len(failed) > 5 else ''}")

        if max_samples is not None:
            samples = samples[:max_samples]

        return samples

    # Dataset interface

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample     = self.samples[idx]
        patient_id = sample["patient_id"]
        label      = sample["label"]
        bmi        = sample["bmi"]
        whr        = sample["whr"]

        if patient_id in self._patient_cache:
            glucose = self._patient_cache[patient_id]
        else:
            glucose = self._load_glucose_array(patient_id)

        mean = glucose.mean()
        std  = glucose.std() if glucose.std() > 0 else 1.0
        glucose_normalised = (glucose - mean) / std

        stats = get_cgm_stats(pd.DataFrame({"glucose": glucose}))

        ts_text = self._build_ts_text(stats, mean, std)

        if self.format_fn is not None:
            ts_text += "\n" + self.format_fn(glucose_normalised)

        pre_prompt  = self._build_pre_prompt(stats, bmi, whr)
        post_prompt = self._build_post_prompt()
        answer      = self._build_answer(patient_id, label, stats, bmi, whr)

        return {
            "pre_prompt":       pre_prompt,
            "post_prompt":      post_prompt,
            "time_series":      [glucose_normalised.tolist()],
            "time_series_text": [ts_text],
            "answer":           answer,
            "patient_id":       patient_id,
            "label":            label,
        }

    # Prompt builders

    def _build_ts_text(self, stats: dict, mean: float, std: float) -> str:
        duration_days = stats["n_readings"] / READINGS_PER_HOUR / 24
        return (
            f"This is a {duration_days:.1f}-day continuous glucose monitoring "
            f"recording from a Dexcom G6 sensor, sampled every 5 minutes "
            f"({stats['n_readings']} readings). "
            f"It has mean {mean:.1f} mg/dL and std {std:.1f} mg/dL. "
            f"The series has been normalised to zero mean and unit variance."
        )

    def _build_pre_prompt(self, stats: dict, bmi: float, whr: float) -> str:
        options = "\n".join(f"  - {label}" for label in LABELS)
        duration_days = stats["n_readings"] / READINGS_PER_HOUR / 24
        return (
            f"You are an expert endocrinologist analysing continuous glucose "
            f"monitoring (CGM) data alongside clinical measurements. Your task "
            f"is to classify a patient's insulin resistance status into one of "
            f"the following categories:\n"
            f"{options}\n\n"
            f"Glucose summary statistics for this {duration_days:.1f}-day recording:\n"
            f"  Mean glucose    : {stats['mean']} mg/dL\n"
            f"  Std deviation   : {stats['std']} mg/dL\n"
            f"  Min             : {stats['min']} mg/dL\n"
            f"  Max             : {stats['max']} mg/dL\n"
            f"  Time in range   : {stats['pct_in_range']}% "
            f"(target 70–180 mg/dL)\n"
            f"  Time below range: {stats['pct_low']}%\n"
            f"  Time above range: {stats['pct_high']}%\n\n"
            f"Clinical measurements:\n"
            f"  BMI                : {bmi:.2f} kg/m²\n"
            f"  Waist-to-Hip Ratio : {whr:.3f}\n\n"
            f"The CGM time series follows:"
        )

    def _build_post_prompt(self) -> str:
        return (
            "Based on this CGM data and clinical measurements, reason step by "
            "step about how the glucose patterns, variability, and adiposity "
            "markers relate to insulin resistance, then provide your classification."
        )

    def _build_answer(
        self, patient_id: str, label: str, stats: dict, bmi: float, whr: float
    ) -> str:
        """
        Build the answer string: chain-of-thought reasoning followed by label.

        Uses a GPT-4o-generated reasoning label if available (loaded from
        labels_path), keyed by person_id. Falls back to a template otherwise.
        """
        if patient_id in self.labels:
            reasoning = self.labels[patient_id]
        else:
            reasoning = self._fallback_reasoning(label, stats, bmi, whr)
        return f"{reasoning}{self.eos_token}"

    def _fallback_reasoning(
        self, label: str, stats: dict, bmi: float, whr: float
    ) -> str:
        duration_days = stats["n_readings"] / READINGS_PER_HOUR / 24
        return (
            f"Examining this {duration_days:.1f}-day CGM trace: mean glucose is "
            f"{stats['mean']} mg/dL with {stats['pct_in_range']}% time in range "
            f"(70–180 mg/dL), {stats['pct_high']}% above range, and "
            f"{stats['pct_low']}% below range. The patient has a BMI of "
            f"{bmi:.1f} kg/m² and a waist-to-hip ratio of {whr:.3f}. "
            f"Based on these glucose dynamics and adiposity markers, the "
            f"patient's insulin resistance status is consistent with {label}.\n"
            f"Answer: {label}"
        )

    # Static helpers

    @staticmethod
    def get_labels() -> List[str]:
        """Return the list of possible class labels. Used by the evaluator."""
        return LABELS
