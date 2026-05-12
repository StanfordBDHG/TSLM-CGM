#
# This source file is part of the TSLM-CGM project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
# Generate pseudo chain-of-thought reasoning labels for insulin resistance
# classification using GPT-4o vision. For each training patient, generates
# a CGM plot and passes it alongside CGM summary statistics, BMI, and
# waist-to-hip ratio to GPT-4o, asking the model to reason about how the
# CGM signal connects to the insulin resistance classification.
#
# Output: a JSON file mapping person_id -> reasoning string, saved to
# cgm_diabetes/ir_prediction/ir_labels.json.
#
# Usage:
#   python -m cgm_diabetes.ir_prediction.generate_ir_labels
#   python -m cgm_diabetes.ir_prediction.generate_ir_labels --split train
#   python -m cgm_diabetes.ir_prediction.generate_ir_labels --max_patients 10
#   python -m cgm_diabetes.ir_prediction.generate_ir_labels --dry_run --max_patients 3
#
# Safe to interrupt and resume — already-generated labels are saved
# incrementally and skipped on subsequent runs.
#

import os
import json
import time
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

from cgm_diabetes.data.cgm_loader import (
    load_cgm_for_patient,
    get_cgm_stats,
    DEFAULT_CACHE_DIR,
)
from cgm_diabetes.captioning.generate_captions import (
    make_cgm_plot,
    encode_image_base64,
    HYPO_THRESHOLD,
    HYPER_THRESHOLD,
)

load_dotenv()

# Output paths
DEFAULT_LABELS_PATH = Path("cgm_diabetes/ir_prediction/ir_labels.json")
DRY_RUN_PLOTS_DIR   = Path("cgm_diabetes/ir_prediction/dry_run_plots")

# Input data
IR_SCORES_CSV = Path("cgm_diabetes/ir_prediction/data/ir_scores.csv")

# OpenAI config
MODEL       = "gpt-4o"
TEMPERATURE = 0.3
MAX_TOKENS  = 1000
RETRY_LIMIT = 3
RETRY_DELAY = 2

# Label display names
LABEL_DESCRIPTIONS = {
    "IR":     "insulin resistant",
    "Non_IR": "not insulin resistant",
}

# Prompt templates
SYSTEM_PROMPT = """You are an expert endocrinologist who specialises in interpreting \
continuous glucose monitoring (CGM) data and assessing metabolic health. Reason carefully \
step by step about both CGM signal patterns and clinical measurements before giving a \
classification, considering glucose variability, glycemic control, and adiposity markers \
in relation to insulin sensitivity."""

USER_PROMPT_TEMPLATE = """A patient wore a Dexcom G6 CGM sensor for {duration_days:.1f} days \
({n_readings} readings at 5-minute intervals).

The attached image shows the patient's full CGM trace. The dashed green line at \
{hypo} mg/dL marks the hypoglycemia threshold and the dashed red line at \
{hyper} mg/dL marks the hyperglycemia threshold. Time in the green band \
(70-180 mg/dL) is the target range.

CGM summary statistics:

  Mean glucose     : {mean} mg/dL
  Std deviation    : {std} mg/dL
  Minimum          : {min} mg/dL
  Maximum          : {max} mg/dL
  Time in range    : {pct_in_range}%  (target 70-180 mg/dL)
  Time below range : {pct_low}%   (hypoglycaemia <{hypo} mg/dL)
  Time above range : {pct_high}%  (hyperglycaemia >{hyper} mg/dL)

Clinical measurements:

  BMI                 : {bmi:.2f} kg/m²
  Waist-to-Hip Ratio  : {whr:.3f}

Please reason step by step about how the CGM signal patterns and clinical measurements \
together inform the insulin resistance classification. Connect CGM features — such as \
glucose variability, time above range, the shape and duration of postprandial excursions, \
and any reactive hypoglycemic episodes — to the underlying physiology of insulin resistance. \
Discuss how the BMI and waist-to-hip ratio reflect adiposity and central fat distribution, \
and explain how these complement the CGM evidence.

Instructions:
- Analyze the data objectively without presuming a particular label.
- Reason carefully and methodically, connecting CGM patterns to the physiology of insulin resistance.
- Write your reasoning as a single, coherent paragraph. Do not use bullet points, lists, \
or section headers.
- Do not reference the plot, visuals, summary statistics, or the process of viewing the data; \
focus only on the characteristics of the time series and clinical measurements.
- Do not mention or speculate about any classification during the rationale, only reveal the \
correct classification at the very end.
- Never state that you are uncertain or unable to classify the data. You must always \
provide a rationale and a final answer.

The confirmed classification for this patient is: {label_description}. Do not mention \
this anywhere in your reasoning.
Your response must end with exactly:
Answer: {label}"""


# Prompt building

def build_prompt(stats: dict, bmi: float, whr: float, label: str) -> str:
    """Fill the user prompt template with CGM stats and patient measurements."""
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
        bmi=bmi,
        whr=whr,
        label_description=LABEL_DESCRIPTIONS[label],
        label=label,
    )


# API call

def call_gpt4_vision(client: OpenAI, user_prompt: str, image_b64: str) -> str:
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

def load_existing_labels(output_path: Path) -> dict:
    """Load previously generated labels so interrupted runs can resume."""
    if output_path.exists():
        with open(output_path) as f:
            labels = json.load(f)
        print(f"[labels] Loaded {len(labels)} existing labels from {output_path}")
        return labels
    return {}


def save_labels(labels: dict, output_path: Path) -> None:
    """Atomically save the labels dict to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(labels, f, indent=2)
    tmp.replace(output_path)   # atomic on POSIX


# Main generation loop

def generate_ir_labels(
    split: Optional[str] = "train",
    output_path: Path = DEFAULT_LABELS_PATH,
    cache_dir: Optional[Path] = None,
    max_patients: Optional[int] = None,
    min_days: float = 8.0,
    dry_run: bool = False,
    force_regenerate: bool = False,
) -> None:
    """
    Generate GPT-4o vision chain-of-thought reasoning labels for insulin
    resistance classification.

    Parameters
    ----------
    split : str or None
        "train", "val", "test", or None for all splits. Defaults to "train".
    output_path : Path
        Where to save the labels JSON.
    cache_dir : Path or None
        CGM parquet cache directory (defaults to DEFAULT_CACHE_DIR).
    max_patients : int or None
        Cap on number of patients — useful for testing cost before full run.
    min_days : float
        Skip patients with fewer than this many days of CGM data.
    dry_run : bool
        If True, generate and save plots and print prompts without calling the API.
    force_regenerate : bool
        If True, ignore existing labels and regenerate all from scratch.
    """
    if not IR_SCORES_CSV.exists():
        raise FileNotFoundError(
            f"IR scores CSV not found: {IR_SCORES_CSV}\n"
            "Run compute_ir_scores.py first to generate it."
        )

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    min_readings = int(min_days * 24 * 12)

    df = pd.read_csv(IR_SCORES_CSV, dtype={"person_id": str})

    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)

    print(f"[labels] {len(df)} patients in split='{split or 'all'}'")

    labels = {} if force_regenerate else load_existing_labels(output_path)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not dry_run:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Add it to your .env file: OPENAI_API_KEY=sk-..."
        )
    client = OpenAI(api_key=api_key) if not dry_run else None

    processed        = 0
    skipped_existing = 0
    skipped_short    = 0
    failed           = []

    for _, row in df.iterrows():
        if max_patients is not None and processed >= max_patients:
            break

        person_id = str(row["person_id"])

        # Skip already-generated
        if person_id in labels:
            skipped_existing += 1
            continue

        ir_label = row["insulin_resistance"]
        bmi      = float(row["bmi"])
        whr      = float(row["waist_to_hip_ratio"])

        if ir_label not in LABEL_DESCRIPTIONS:
            print(f"  [warning] {person_id}: unknown label '{ir_label}' — skipping")
            failed.append(person_id)
            continue

        # Load CGM data
        try:
            cgm_df = load_cgm_for_patient(person_id, cache_dir=cache_dir)
        except Exception as e:
            print(f"  [error] {person_id}: could not load CGM — {e}")
            failed.append(person_id)
            continue

        # Skip short recordings
        if len(cgm_df) < min_readings:
            skipped_short += 1
            continue

        stats  = get_cgm_stats(cgm_df)
        prompt = build_prompt(stats, bmi, whr, ir_label)

        # Generate plot
        try:
            image_bytes = make_cgm_plot(cgm_df, person_id)
            image_b64   = encode_image_base64(image_bytes)
        except Exception as e:
            print(f"  [error] {person_id}: could not generate plot — {e}")
            failed.append(person_id)
            continue

        if dry_run:
            print(f"\n{'='*60}")
            print(f"Patient {person_id} ({ir_label})  BMI={bmi:.1f}  WHR={whr:.3f}")
            print(f"{'='*60}")
            plot_path = DRY_RUN_PLOTS_DIR / f"{person_id}.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plot_path.write_bytes(image_bytes)
            print(f"[plot] Saved to {plot_path}  ({len(image_bytes) / 1024:.1f} KB)")
            print(prompt)
            processed += 1
            continue

        # Call GPT-4o vision
        try:
            print(f"  [{processed + 1}] Patient {person_id} ({ir_label}) "
                  f"BMI={bmi:.1f} WHR={whr:.3f} — {stats['n_readings']} readings ...")
            label_text = call_gpt4_vision(client, prompt, image_b64)

            labels[person_id] = label_text
            processed += 1

            # Save after every patient so no progress is ever lost
            save_labels(labels, output_path)
            print(f"    ✓ Saved ({len(labels)} total)")

        except Exception as e:
            print(f"API failed for {person_id}: {e}")
            failed.append(person_id)

    # Summary
    print("\n[labels] Done.")
    print(f"  Generated                 : {processed}")
    print(f"  Skipped (already existed) : {skipped_existing}")
    print(f"  Skipped (too short)       : {skipped_short}")
    print(f"  Failed                    : {len(failed)}")
    if failed:
        print(f"  Failed IDs: {failed[:10]}"
              f"{'...' if len(failed) > 10 else ''}")
    print(f"  Total labels in file      : {len(labels)}")
    if not dry_run:
        print(f"  Output: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GPT-4o vision CoT reasoning labels for IR classification."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Split to generate labels for (default: train)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help=f"Output JSON path (default: {DEFAULT_LABELS_PATH})",
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
        help="Minimum CGM recording length in days (default: 8.0)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Generate plots and print prompts without calling the API",
    )
    parser.add_argument(
        "--force_regenerate",
        action="store_true",
        help="Ignore existing labels and regenerate all from scratch",
    )
    args = parser.parse_args()

    generate_ir_labels(
        split=args.split,
        output_path=args.output,
        max_patients=args.max_patients,
        min_days=args.min_days,
        dry_run=args.dry_run,
        force_regenerate=args.force_regenerate,
    )


if __name__ == "__main__":
    main()
