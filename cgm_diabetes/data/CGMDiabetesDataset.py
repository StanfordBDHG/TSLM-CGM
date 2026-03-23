#
# This source file is part of the TSLM-CGM project
#

import os
import json
from typing import Optional, Callable, List, Dict
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from cgm_diabetes.data.participants import load_participants
from cgm_diabetes.data.cgm_loader import (
    load_cgm_for_patient,
    prefetch_all_cgm,
    get_cgm_stats,
    DEFAULT_CACHE_DIR,
)

load_dotenv()

# Window size configuration
# Change WINDOW_HOURS to use a different window size, e.g. WINDOW_HOURS = 72 for 3 days, 168 for 7 days
WINDOW_HOURS = 24
READINGS_PER_HOUR = 12 # Dexcom G6: one reading every 5 minutes
WINDOW_SIZE = WINDOW_HOURS * READINGS_PER_HOUR # = 288 for 24 hours
STEP_SIZE = WINDOW_SIZE # Non-overlapping windows (one per day), set to WINDOW_SIZE // 2 for 50% overlap

LABELS = [
    "healthy",
    "prediabetes_lifestyle",
    "oral_non_insulin",
    "insulin_dependent",
]


class CGMDiabetesDataset(Dataset):
    """
    PyTorch Dataset for CGM-based diabetes classification.
 
    Each sample is a sliding window of WINDOW_SIZE glucose readings (default
    288 = 24 hours at 5-minute intervals) from a single patient, paired with
    a chain-of-thought caption and a diabetes category label.
 
    Data caching:
    On the first call for a given patient, data is downloaded from Azure
    Blob Storage and saved to ``cache_dir/cgm/<patient_id>.parquet``.
    Subsequent calls laod from disk.
 
    To pre-download everything before training starts:
        CGMDiabetesDataset.prefetch(split="train")  # or "val" / "test" / None for all
 
    Parameters:
    split : str
        One of "train", "val", "test".
    EOS_TOKEN : str
        End-of-sequence token string from the model tokenizer.
    window_hours : int
        Length of each sample window in hours. Default 24.
    step_hours : int or None
        How far to advance the window between samples, in hours.
        None (default) means non-overlapping (step = window size).
    max_samples : int or None
        Optional cap for total number of samples, used for testing.
    captions_path : str or None
        Path to JSON file mapping patient_id -> caption string.
        If None, simple statistical summary is used instead.
        Generate captions with cgm_diabetes/captioning/generate_captions.py.
    time_series_format_function : Callable or None
        Optional function to format the raw glucose array into a string.
    format_sample_str : bool
        Whether to format the sample as a string (always True for training).
    cache_dir : Path or None
        Root directory for local data cache.
        Defaults to DEFAULT_CACHE_DIR
    force_download : bool
        If True, re-download from Azure even if a cache file exists.
    """
 
    def __init__(
        self,
        split: str = "train",
        EOS_TOKEN: str = "",
        window_hours: int = WINDOW_HOURS,
        step_hours: Optional[int] = None,
        max_samples: Optional[int] = None,
        captions_path: Optional[str] = None,
        time_series_format_function: Optional[Callable] = None,
        format_sample_str: bool = True,
        cache_dir: Optional[Path] = None,
        force_download: bool = False,
    ):
        super().__init__()
 
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"
 
        self.split          = split
        self.eos_token      = EOS_TOKEN
        self.format_fn      = time_series_format_function
        self.cache_dir      = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.force_download = force_download
 
        # Window
        self.window_hours = window_hours
        self.window_size  = window_hours * READINGS_PER_HOUR
        self.step_size    = (step_hours or window_hours) * READINGS_PER_HOUR
 
        # Load captions if provided
        self.captions: Dict[str, str] = {}
        if captions_path is not None:
            with open(captions_path) as f:
                self.captions = json.load(f)
            print(f"[CGMDiabetesDataset] Loaded {len(self.captions)} captions "
                  f"from {captions_path}")
 
        print(f"[CGMDiabetesDataset] Building {split} sample index...")
        self._patient_cache: Dict[str, np.ndarray] = {}  # in-process cache
        self.samples = self._build_sample_index(split, max_samples)
        print(f"[CGMDiabetesDataset] {len(self.samples)} windows across "
              f"{len(set(s['patient_id'] for s in self.samples))} patients")
 

    # Prefetch all data for a split up front
 
    @classmethod
    def prefetch(
        cls,
        split: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        force_download: bool = False,
    ) -> None:
        """
        Download and cache CGM data for all patients in a split (or all splits).
 
        Call once before training to avoid Azure latency during the
        first epoch. Subsequent dataset instantiations load from the local parquet files.
        """
        participants = load_participants()
        if split is not None:
            participants = {
                pid: info for pid, info in participants.items()
                if info["split"] == split
            }
        prefetch_all_cgm(
            list(participants.keys()),
            cache_dir=cache_dir,
            force_download=force_download,
        )
 

    # Index building
 
    def _load_glucose_array(self, patient_id: str) -> np.ndarray:
        """Load glucose values for one patient, using the local file cache."""
        df = load_cgm_for_patient(
            patient_id,
            cache_dir=self.cache_dir,
            force_download=self.force_download,
        )
        return df["glucose"].to_numpy(dtype=np.float32)
 
    def _build_sample_index(
        self, split: str, max_samples: Optional[int]
    ) -> List[Dict]:
        """
        For each patient in the split, load their CGM data (from disk cache or
        Azure) and record the (start, end) slice indices for each window.
        """
        participants = load_participants()
        split_patients = {
            pid: info for pid, info in participants.items()
            if info["split"] == split
        }
 
        samples = []
        failed = []
 
        for patient_id, info in split_patients.items():
            if max_samples is not None and len(samples) >= max_samples:
                break
            try:
                glucose = self._load_glucose_array(patient_id)
 
                if len(glucose) < self.window_size:
                    print(f"  [skip] {patient_id}: only {len(glucose)} readings "
                          f"(need {self.window_size})")
                    continue
 
                # Cache the array in-process to avoid re-reading parquet for
                # every window of the same patient during index build
                self._patient_cache[patient_id] = glucose
 
                start = 0
                while start + self.window_size <= len(glucose):
                    samples.append({
                        "patient_id": patient_id,
                        "label":      info["label"],
                        "start":      start,
                        "end":        start + self.window_size,
                    })
                    start += self.step_size
 
            except Exception as e:
                failed.append(patient_id)
                print(f"  [error] {patient_id}: {e}")
 
        if failed:
            print(f"[CGMDiabetesDataset] Failed to load {len(failed)} patients: "
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
        start      = sample["start"]
        end        = sample["end"]
 
        # Load glucose window
        if patient_id in self._patient_cache:
            glucose = self._patient_cache[patient_id][start:end]
        else:
            glucose = self._load_glucose_array(patient_id)[start:end]
 
        # Normalize to 0 mean / unit variance for the model
        # Raw values are preserved in stats for the text prompt
        mean = glucose.mean()
        std  = glucose.std() if glucose.std() > 0 else 1.0
        glucose_normalised = (glucose - mean) / std
 
        # Statistics for prompt
        stats = get_cgm_stats(pd.DataFrame({"glucose": glucose}))
 
        # Time series text (description inserted between pre/post prompt)
        ts_text = self._build_ts_text(stats, mean, std)
 
        # Optionally append formatted glucose values (Gruver-style baseline)
        if self.format_fn is not None:
            ts_text += "\n" + self.format_fn(glucose_normalised)
 
        pre_prompt  = self._build_pre_prompt(stats)
        post_prompt = self._build_post_prompt()
        answer      = self._build_answer(patient_id, label, stats)
 
        return {
            # Required by OpenTSLM training loop
            "pre_prompt":       pre_prompt,
            "post_prompt":      post_prompt,
            "time_series":      [glucose_normalised.tolist()],
            "time_series_text": [ts_text],
            "answer":           answer,
            # Extra fields for evaluation and caption generation
            "patient_id":       patient_id,
            "label":            label,
        }
 

    # Prompt builders
 
    def _build_ts_text(self, stats: dict, mean: float, std: float) -> str:
        return (
            f"This is a {self.window_hours}-hour continuous glucose monitoring "
            f"recording from a Dexcom G6 sensor, sampled every 5 minutes "
            f"({stats['n_readings']} readings). "
            f"It has mean {mean:.1f} mg/dL and std {std:.1f} mg/dL. "
            f"The series has been normalised to zero mean and unit variance."
        )
 
    def _build_pre_prompt(self, stats: dict) -> str:
        options = "\n".join(f"  - {label}" for label in LABELS)
        return (
            f"You are an expert endocrinologist analysing continuous glucose "
            f"monitoring (CGM) data. Your task is to classify a patient's "
            f"diabetes status into one of the following categories:\n"
            f"{options}\n\n"
            f"Glucose summary statistics for this {self.window_hours}-hour window:\n"
            f"  Mean glucose    : {stats['mean']} mg/dL\n"
            f"  Std deviation   : {stats['std']} mg/dL\n"
            f"  Min             : {stats['min']} mg/dL\n"
            f"  Max             : {stats['max']} mg/dL\n"
            f"  Time in range   : {stats['pct_in_range']}% "
            f"(target 70–180 mg/dL)\n"
            f"  Time below range: {stats['pct_low']}%\n"
            f"  Time above range: {stats['pct_high']}%\n\n"
            f"The CGM time series follows:"
        )
 
    def _build_post_prompt(self) -> str:
        return (
            "Based on this continuous glucose monitoring data, reason step by "
            "step about the patient's glucose patterns, then provide your "
            "diagnosis."
        )
 
    def _build_answer(self, patient_id: str, label: str, stats: dict) -> str:
        """
        Build the answer string: chain-of-thought reasoning followed by label.
 
        Uses a GPT-4-generated caption if available (loaded from captions_path),
        keyed by patient_id. Falls back to a template otherwise.
        """
        if patient_id in self.captions:
            reasoning = self.captions[patient_id]
        else:
            reasoning = self._fallback_reasoning(label, stats)
        return f"{reasoning}\nAnswer: {label}{self.eos_token}"
 
    def _fallback_reasoning(self, label: str, stats: dict) -> str:
        """
        Template-based reasoning used before GPT-4 captions are generated.
        Replace with generate_captions.py output before final training.
        """
        return (
            f"Examining this {self.window_hours}-hour CGM trace: the mean "
            f"glucose is {stats['mean']} mg/dL with {stats['pct_in_range']}% "
            f"time in range (70–180 mg/dL), {stats['pct_high']}% time above "
            f"range, and {stats['pct_low']}% time below range. "
            f"Based on these glucose dynamics, the patient's diabetes status "
            f"is consistent with {label}."
        )
 
    # Static helpers
 
    @staticmethod
    def get_labels() -> List[str]:
        """Return the list of possible class labels. Used by the evaluator."""
        return LABELS