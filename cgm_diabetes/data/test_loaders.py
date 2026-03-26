#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
# Verify participants.py, cgm_loader.py, and CGMDiabetesDataset.py are working.
#

import time
from pathlib import Path
from cgm_diabetes.data.participants import load_participants, get_label_distribution
from cgm_diabetes.data.cgm_loader import (
    load_cgm_for_patient,
    get_cgm_stats,
    DEFAULT_CACHE_DIR,
)
# Test participants
print("=== Participants ===")
participants = load_participants()
print(f"Label distribution: {get_label_distribution(participants)}")
print(f"First 3 entries:")
for pid, info in list(participants.items())[:3]:
    print(f"  {pid}: {info}")

# Test CGM Loader and Cache
print("\n=== CGM Loader + Cache ===")
 
TEST_PATIENT = "1001"
cache_file = DEFAULT_CACHE_DIR / "cgm" / f"{TEST_PATIENT}.parquet"
 
# Show where cache lives
print(f"Cache directory : {DEFAULT_CACHE_DIR.resolve()}")
print(f"Cache file path : {cache_file}")
print(f"Cache exists    : {cache_file.exists()}")
 
# First load — downloads from Azure if cache missing, otherwise reads from disk
print(f"\n--- First load (patient {TEST_PATIENT}) ---")
t0 = time.perf_counter()
df = load_cgm_for_patient(TEST_PATIENT)
t1 = time.perf_counter()
first_load_ms = (t1 - t0) * 1000
print(f"Load time       : {first_load_ms:.1f} ms")
print(f"Cache now exists: {cache_file.exists()}")
if cache_file.exists():
    size_kb = cache_file.stat().st_size / 1024
    print(f"Cache file size : {size_kb:.1f} KB")
print(f"Readings        : {len(df)}")
print(f"Columns         : {list(df.columns)}")
print(f"First 5 timestamps : {df['timestamp'][:5].tolist()}")
print(f"First 5 glucose    : {df['glucose'][:5].tolist()}")
print(f"Stats: {get_cgm_stats(df)}")
 
# Second load — must come from cache (should be much faster than first)
print(f"\n--- Second load (patient {TEST_PATIENT}, should be cached) ---")
t0 = time.perf_counter()
df2 = load_cgm_for_patient(TEST_PATIENT)
t1 = time.perf_counter()
second_load_ms = (t1 - t0) * 1000
print(f"Load time       : {second_load_ms:.1f} ms")
speedup = first_load_ms / second_load_ms if second_load_ms > 0 else float("inf")
print(f"Speedup vs first: {speedup:.1f}x")
 
# Verify the two loads returned identical data
assert df.equals(df2), "ERROR: cached data does not match original download!"
print("Data integrity  : OK (cached data matches original)")
 
# Force re-download and verify it still matches
print(f"\n--- Force re-download (force_download=True) ---")
t0 = time.perf_counter()
df3 = load_cgm_for_patient(TEST_PATIENT, force_download=True)
t1 = time.perf_counter()
print(f"Load time       : {(t1 - t0) * 1000:.1f} ms")
assert df.equals(df3), "ERROR: force-downloaded data does not match cached data!"
print("Data integrity  : OK (re-download matches cache)")
 
# Test Dataset
print("\n=== CGMDiabetesDataset ===")
from cgm_diabetes.data.CGMDiabetesDataset import CGMDiabetesDataset
 
dataset = CGMDiabetesDataset(split="train", EOS_TOKEN="", max_samples=5)
print(f"Dataset size: {len(dataset)}")
sample = dataset[0]
print(f"Keys: {list(sample.keys())}")
print(f"Pre-prompt:\n{sample['pre_prompt']}")
print(f"Post-prompt:\n{sample['post_prompt']}")
print(f"Time series length: {len(sample['time_series'][0])}")
print(f"First 5 time series values: {sample['time_series'][0][:5]}")
print(f"Time series text: {sample['time_series_text'][0]}")
print(f"Answer:\n{sample['answer']}")
print(f"Patient ID: {sample['patient_id']}")
print(f"Label: {sample['label']}")
 
# Verify __getitem__ is fast on repeated access (hitting _patient_cache)
print(f"\n--- __getitem__ speed (in-process cache) ---")
t0 = time.perf_counter()
for i in range(len(dataset)):
    _ = dataset[i]
t1 = time.perf_counter()
per_item_ms = (t1 - t0) * 1000 / len(dataset)
print(f"Avg per __getitem__: {per_item_ms:.2f} ms  ({len(dataset)} samples)")
 