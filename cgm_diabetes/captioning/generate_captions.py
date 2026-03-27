#
# This source file is part of the TSLM-CGM project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
# Generate chain-of-thought captions for CGM patients using GPT-4o vision.
# For each patient, generates a CGM plot with hypo/hyperglycemia reference
# lines and passes it alongside summary statistics to GPT-4o.
#
# Output: a JSON file mapping patient_id -> caption string, compatible
# with CGMDiabetesDataset(captions_path=...).
#
# Usage:
#   python -m cgm_diabetes.captioning.generate_captions
#   python -m cgm_diabetes.captioning.generate_captions --split train
#   python -m cgm_diabetes.captioning.generate_captions --split train --max_patients 10
#   python -m cgm_diabetes.captioning.generate_captions --dry_run --max_patients 3
#
# Safe to interrupt and resume — already-generated captions
# are saved incrementally and skipped on subsequent runs.
#

import os
import io
import json
import time
import base64
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for scripts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from dotenv import load_dotenv
from openai import OpenAI

from cgm_diabetes.data.participants import load_participants
from cgm_diabetes.data.cgm_loader import (
    load_cgm_for_patient,
    get_cgm_stats,
    DEFAULT_CACHE_DIR,
)

load_dotenv()

# Output path
DEFAULT_CAPTIONS_PATH = Path("cgm_diabetes/captioning/captions.json")

# OpenAI config
MODEL       = "gpt-4o"       # Vision-capable model
TEMPERATURE = 0.3            # Matches temperature used for ECG and activity captions
MAX_TOKENS  = 1000
RETRY_LIMIT = 3
RETRY_DELAY = 2

# CGM clinical thresholds 
HYPO_THRESHOLD  = 70 
HYPER_THRESHOLD = 180

# Label display names 
LABEL_DESCRIPTIONS = {
    "healthy":               "no diabetes",
    "prediabetes_lifestyle": "prediabetes managed with lifestyle changes only",
    "oral_non_insulin":      "diabetes managed with oral or non-insulin injectable medication",
    "insulin_dependent":     "insulin-dependent diabetes",
}

# Prompt templates
SYSTEM_PROMPT = """You are an expert endocrinologist who specialises in interpreting \
continuous glucose monitoring (CGM) data. Reason carefully step by step before giving \
a diagnosis, considering glucose variability and overall glycemic control."""

USER_PROMPT_TEMPLATE = """A patient wore a Dexcom G6 CGM sensor for {duration_days:.1f} days \
({n_readings} readings at 5-minute intervals).

The attached image shows the patient's full CGM trace. The dashed green line at \
{hypo} mg/dL marks the hypoglycemia threshold and the dashed red line at \
{hyper} mg/dL marks the hyperglycemia threshold. Time in the green band \
(70-180 mg/dL) is the target range.

Summary statistics for the full recording:

  Mean glucose     : {mean} mg/dL
  Std deviation    : {std} mg/dL
  Minimum          : {min} mg/dL
  Maximum          : {max} mg/dL
  Time in range    : {pct_in_range}%  (target 70-180 mg/dL)
  Time below range : {pct_low}%   (hypoglycaemia <{hypo} mg/dL)
  Time above range : {pct_high}%  (hyperglycaemia >{hyper} mg/dL)

Please reason step by step about what this CGM trace and these statistics reveal \
about the patient's glucose regulation, glycemic variability, and overall \
metabolic control. Reference specific visual features of the trace where relevant \
(e.g. nocturnal patterns, hypoglycemic episodes). \
Then state the diagnosis.

Your task is to determine the correct diabetes category based solely on the observed patterns in \
the time series. \

Instructions: \
- Analyze the data objectively without presuming a particular label. \
- Reason carefully and methodically about what the signal patterns suggest about glucose control \
- Write your reasoning as a single, coherent paragraph. Do not use bullet points, lists, \
or section headers. \
- Do not reference the plot, visuals, summary statistics, or the process of viewing the data in your \
explanation; focus only on the characteristics of the time series. \
- Do not mention or speculate about any class during the rationale, only reveal the \
correct class at the very end. \
- Never state that you are uncertain or unable to classify the data. You must always \
provide a rationale and a final answer. \

The confirmed diagnosis for this patient is: {label_description}. Do not mention this anywhere in \
your reasoning. \
Your response must end with exactly: \
Answer: {label}"""


# Plot generation

def make_cgm_plot(df: pd.DataFrame, patient_id: str) -> bytes:
    """
    Render the full CGM trace for a patient as a PNG image.

    The plot includes:
    - Glucose trace coloured by zone (blue = in range, red = high, green = low)
    - Dashed reference lines at 70 mg/dL (hypo) and 180 mg/dL (hyper)
    - Shaded target range band (70-180 mg/dL)
    - One subplot per day 
    - Gaps where sensor dropout exceeds 10 mins

    Returns the PNG as raw bytes (ready for base64 encoding).
    """
    # Normalise timestamps to date only (calendar day in the sensor's local date).
    # Timestamps are UTC; using .dt.date gives consistent day boundaries.
    ts = pd.to_datetime(df["timestamp"], utc=True)
    dates = ts.dt.date
    unique_days = sorted(dates.unique())
    n_days = max(1, len(unique_days))

    fig, axes = plt.subplots(
        n_days, 1,
        figsize=(14, 2.2 * n_days),
        sharex=False,
    )
    if n_days == 1:
        axes = [axes]

    fig.suptitle(
        f"Full CGM Recording ({n_days} days)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    max_gap = pd.Timedelta(minutes=10)

    for day_idx, (ax, day) in enumerate(zip(axes, unique_days)):
        mask = dates == day

        # Use a reset index so iloc lookups are contiguous
        day_ts = ts[mask].reset_index(drop=True)

        g = df.loc[mask, "glucose"].to_numpy()
        # Hour-of-day derived from real timestamps (fractional hours)
        t = (
            day_ts.dt.hour
            + day_ts.dt.minute / 60
            + day_ts.dt.second / 3600
        ).to_numpy()

        # Shaded target band
        ax.axhspan(HYPO_THRESHOLD, HYPER_THRESHOLD, color="#e8f5e9", alpha=0.5, zorder=0)

        # Reference lines
        ax.axhline(HYPO_THRESHOLD,  color="#2e7d32", linestyle="--",
                   linewidth=1.2, label=f"Hypo  ({HYPO_THRESHOLD} mg/dL)",  zorder=2)
        ax.axhline(HYPER_THRESHOLD, color="#c62828", linestyle="--",
                   linewidth=1.2, label=f"Hyper ({HYPER_THRESHOLD} mg/dL)", zorder=2)

        # Glucose trace — colour segments by zone, skip across sensor dropout gaps
        for i in range(len(g) - 1):
            if day_ts.iloc[i + 1] - day_ts.iloc[i] > max_gap:
                continue  # gap too large — don't draw a line across dropout
            x_seg = [t[i], t[i + 1]]
            y_seg = [g[i], g[i + 1]]
            mid   = (g[i] + g[i + 1]) / 2
            if mid < HYPO_THRESHOLD:
                colour = "#1b5e20"    # dark green — hypoglycaemia
            elif mid > HYPER_THRESHOLD:
                colour = "#b71c1c"    # dark red — hyperglycaemia
            else:
                colour = "#1565c0"    # blue — in range
            ax.plot(x_seg, y_seg, color=colour, linewidth=1.0, zorder=3)


        # Y-axis limits with padding
        y_min = max(0,   g.min() - 20) if len(g) else 0
        y_max = min(450, g.max() + 20) if len(g) else 450
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, 24)

        ax.set_ylabel("Glucose\n(mg/dL)", fontsize=8)
        ax.set_xlabel("Hour of day", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"Day {day_idx + 1}", fontsize=9, loc="left", pad=2)
        ax.set_xticks(range(0, 25, 4))
        ax.grid(axis="y", alpha=0.3, zorder=1)

        # Only add legend to first subplot
        if day_idx == 0:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.8)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def encode_image_base64(image_bytes: bytes) -> str:
    """Encode raw image bytes as a base64 string for the OpenAI API."""
    return base64.b64encode(image_bytes).decode("utf-8")


# Prompt building

def build_prompt(stats: dict, label: str) -> str:
    """Fill the user prompt template with patient statistics."""
    duration_days = stats["n_readings"] / 12 / 24
    return USER_PROMPT_TEMPLATE.format(
        duration_days=duration_days,
        n_readings=stats["n_readings"],
        mean=stats["mean"],
        std=stats["std"],
        min=stats["min"],
        max=stats["max"],
        pct_in_range=stats["pct_in_range"],
        pct_low=stats["pct_low"],
        pct_high=stats["pct_high"],
        hypo=HYPO_THRESHOLD,
        hyper=HYPER_THRESHOLD,
        label_description=LABEL_DESCRIPTIONS[label],
        label=label,
    )


# API call

def call_gpt4_vision(
    client: OpenAI,
    user_prompt: str,
    image_b64: str,
) -> str:
    """
    Call GPT-4o with a text prompt and a base64-encoded PNG image.
    Retries up to RETRY_LIMIT times on failure.
    """
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": user_prompt,
                            },
                        ],
                    },
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == RETRY_LIMIT:
                raise
            print(f"    [retry {attempt}/{RETRY_LIMIT}] API error: {e} "
                  f"— waiting {RETRY_DELAY}s")
            time.sleep(RETRY_DELAY)


# Persistence helpers

def load_existing_captions(output_path: Path) -> dict:
    """Load previously generated captions so interrupted runs can resume."""
    if output_path.exists():
        with open(output_path) as f:
            captions = json.load(f)
        print(f"[captions] Loaded {len(captions)} existing captions "
              f"from {output_path}")
        return captions
    return {}


def save_captions(captions: dict, output_path: Path) -> None:
    """Atomically save the captions dict to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(captions, f, indent=2)
    tmp.replace(output_path)   # atomic on POSIX


# Main generation loop

def generate_captions(
    split: Optional[str] = None,
    output_path: Path = DEFAULT_CAPTIONS_PATH,
    cache_dir: Optional[Path] = None,
    max_patients: Optional[int] = None,
    min_days: float = 8.0,
    dry_run: bool = False,
    force_regenerate: bool = False,
) -> None:
    """
    Generate GPT-4o vision captions for all patients in a split (or all splits).

    Parameters
    split : str or None
        "train", "val", "test", or None for all splits.
    output_path : Path
        Where to save the captions JSON.
    cache_dir : Path or None
        CGM parquet cache directory (defaults to DEFAULT_CACHE_DIR).
    max_patients : int or None
        Cap on number of patients — useful for testing cost before full run.
    min_days : float
        Skip patients with fewer than this many days of CGM data.
    dry_run : bool
        If True, generate and save plots but do not call the API.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    min_readings = int(min_days * 24 * 12)

    participants = load_participants()
    if split is not None:
        participants = {
            pid: info for pid, info in participants.items()
            if info["split"] == split
        }
    print(f"[captions] {len(participants)} patients in split='{split or 'all'}'")

    captions = {} if force_regenerate else load_existing_captions(output_path)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not dry_run:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Add it to your .env file: OPENAI_API_KEY=sk-..."
        )
    client = OpenAI(api_key=api_key) if not dry_run else None

    processed       = 0
    skipped_existing = 0
    skipped_short   = 0
    failed          = []

    for patient_id, info in participants.items():
        if max_patients is not None and processed >= max_patients:
            break

        # Skip already-generated
        if patient_id in captions:
            skipped_existing += 1
            continue

        # Load CGM data from parquet cache
        try:
            df = load_cgm_for_patient(patient_id, cache_dir=cache_dir)
        except Exception as e:
            print(f"  [error] {patient_id}: could not load CGM — {e}")
            failed.append(patient_id)
            continue

        # Skip short recordings
        if len(df) < min_readings:
            skipped_short += 1
            continue

        label  = info["label"]
        stats  = get_cgm_stats(df)
        prompt = build_prompt(stats, label)

        # Generate plot
        try:
            image_bytes = make_cgm_plot(df, patient_id)
            image_b64   = encode_image_base64(image_bytes)
        except Exception as e:
            print(f"  [error] {patient_id}: could not generate plot — {e}")
            failed.append(patient_id)
            continue

        if dry_run:
            print(f"\n{'='*60}")
            print(f"Patient {patient_id} ({label})")
            print(f"{'='*60}")
            plot_path = Path("cgm_diabetes/captioning/dry_run_plots") / f"{patient_id}.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plot_path.write_bytes(image_bytes)
            print(f"[plot] Saved to {plot_path}  ({len(image_bytes) / 1024:.1f} KB)")
            print(prompt)
            processed += 1
            continue


        # Call GPT-4o vision
        try:
            print(f"  [{processed + 1}] Patient {patient_id} ({label}) "
                  f"— {stats['n_readings']} readings...")
            caption = call_gpt4_vision(client, prompt, image_b64)

            captions[patient_id] = caption
            processed += 1

            # Save after every patient so no progress is ever lost
            save_captions(captions, output_path)
            print(f"    ✓ Saved ({len(captions)} total)")

        except Exception as e:
            print(f"API failed for {patient_id}: {e}")
            failed.append(patient_id)

    # Summary
    print("\n[captions] Done.")
    print(f"  Generated                 : {processed}")
    print(f"  Skipped (already existed) : {skipped_existing}")
    print(f"  Skipped (too short)       : {skipped_short}")
    print(f"  Failed                    : {len(failed)}")
    if failed:
        print(f"  Failed IDs: {failed[:10]}"
              f"{'...' if len(failed) > 10 else ''}")
    print(f"  Total captions in file    : {len(captions)}")
    if not dry_run:
        print(f"  Output: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GPT-4o vision CoT captions for CGM patients."
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Split to generate captions for (default: all splits)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CAPTIONS_PATH,
        help=f"Output JSON path (default: {DEFAULT_CAPTIONS_PATH})",
    )
    parser.add_argument(
        "--max_patients",
        type=int,
        default=None,
        help="Limit number of patients, useful for cost estimation",
    )
    parser.add_argument(
        "--min_days",
        type=float,
        default=8.0,
        help="Minimum recording length in days (default: 8.0)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Generate plots and print prompts without calling the API",
    )
    parser.add_argument(
        "--force_regenerate",
        action="store_true",
        help="Ignore existing captions and regenerate all from scratch",
    )
    args = parser.parse_args()

    generate_captions(
        split=args.split,
        output_path=args.output,
        max_patients=args.max_patients,
        min_days=args.min_days,
        dry_run=args.dry_run,
        force_regenerate=args.force_regenerate,
    )


if __name__ == "__main__":
    main()