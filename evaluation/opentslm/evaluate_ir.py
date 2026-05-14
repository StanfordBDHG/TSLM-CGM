#!/usr/bin/env python3
#
# This source file is part of the TSLM-CGM project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
# Evaluate TSLM models on CGM insulin resistance (IR) classification.
#
# Runs inference on the IR test set and computes AUC, accuracy, F1, etc.
#
# Usage:
#   # Flamingo pretrained
#   python -m evaluation.opentslm.evaluate_ir \
#       --model_type flamingo --checkpoint hf_models/flamingo_3b_stage5.pt \
#       --output results_ir/flamingo_pretrained.jsonl
#
#   # Flamingo baseline (no fine-tuning)
#   python -m evaluation.opentslm.evaluate_ir \
#       --model_type flamingo \
#       --output results_ir/flamingo_baseline.jsonl
#
#   # SP pretrained
#   python -m evaluation.opentslm.evaluate_ir \
#       --model_type sp --checkpoint hf_models/sp_3b_stage5.pt \
#       --output results_ir/sp_pretrained.jsonl
#
#   # SP baseline
#   python -m evaluation.opentslm.evaluate_ir \
#       --model_type sp \
#       --output results_ir/sp_baseline.jsonl

import sys
import os
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from model.llm.OpenTSLMSP import OpenTSLMSP
from cgm_diabetes.ir_prediction.CGMIRDataset import CGMIRDataset, LABELS
from prompt.full_prompt import FullPrompt
from prompt.text_prompt import TextPrompt
from prompt.text_time_series_prompt import TextTimeSeriesPrompt

load_dotenv()


def load_model(model_type: str, checkpoint_path: str | None, device: str, llm_id: str):
    print(f"Loading {model_type} model (llm: {llm_id})...")
    if model_type == "flamingo":
        model = OpenTSLMFlamingo(
            device=device,
            llm_id=llm_id,
            cross_attn_every_n_layers=1,
        )
    else:
        model = OpenTSLMSP(
            device=device,
            llm_id=llm_id,
        )

    if checkpoint_path:
        print(f"  Loading checkpoint: {checkpoint_path}")
        model.load_from_file(checkpoint_path)
    else:
        print("  No checkpoint — using base LLM weights (baseline)")
        model.to(device)

    model.eval()
    print("Model ready.")
    return model


def extract_ir_label(prediction: str) -> str:
    """Extract IR or Non_IR from model free-text output."""
    pred_lower = prediction.lower()

    # Prefer the explicit "Answer:" marker used in training
    if "answer:" in pred_lower:
        answer_part = pred_lower.split("answer:")[-1].strip()
        if "non_ir" in answer_part or "non-ir" in answer_part:
            return "Non_IR"
        if answer_part.startswith("ir"):
            return "IR"

    # Fallback: scan full output
    if "non_ir" in pred_lower or "non-ir" in pred_lower:
        return "Non_IR"
    if "insulin resistance" in pred_lower and "not" not in pred_lower.split("insulin")[0][-20:]:
        return "IR"

    return "unknown"


def run_evaluation(
    model,
    dataset: CGMIRDataset,
    max_new_tokens: int,
    output_path: str,
) -> list[dict]:
    results = []

    # Resume from partial output if interrupted
    seen_ids: set[str] = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                results.append(r)
                seen_ids.add(r["patient_id"])
        print(f"Resuming — {len(seen_ids)} predictions already saved.")

    out_file = open(output_path, "a")

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            patient_id = sample["patient_id"]
            if patient_id in seen_ids:
                continue

            print(f"[{i + 1}/{len(dataset)}] patient {patient_id}  gt={sample['label']}")

            try:
                pre_prompt = TextPrompt(sample["pre_prompt"])
                post_prompt = TextPrompt(sample["post_prompt"])
                ts_prompts = [
                    TextTimeSeriesPrompt(txt, ts)
                    for txt, ts in zip(sample["time_series_text"], sample["time_series"])
                ]
                prompt = FullPrompt(pre_prompt, ts_prompts, post_prompt)

                prediction = model.eval_prompt(prompt, max_new_tokens=max_new_tokens)
                predicted_label = extract_ir_label(prediction)

                result = {
                    "patient_id": patient_id,
                    "ground_truth": sample["label"],
                    "predicted": predicted_label,
                    "full_prediction": prediction,
                }
                results.append(result)
                out_file.write(json.dumps(result) + "\n")
                out_file.flush()

                print(f"  pred={predicted_label}")

            except Exception as e:
                print(f"  ERROR: {e}")

    out_file.close()
    return results


def compute_metrics(results: list[dict]) -> dict:
    valid = [r for r in results if r["predicted"] != "unknown"]
    n_unknown = len(results) - len(valid)

    if not valid:
        print("No valid predictions — cannot compute metrics.")
        return {"n_total": len(results), "n_valid": 0, "n_unknown": n_unknown}

    gt = [r["ground_truth"] for r in valid]
    pred = [r["predicted"] for r in valid]

    gt_bin = [1 if g == "IR" else 0 for g in gt]
    pred_bin = [1 if p == "IR" else 0 for p in pred]

    metrics = {
        "n_total": len(results),
        "n_valid": len(valid),
        "n_unknown": n_unknown,
        "accuracy": accuracy_score(gt_bin, pred_bin),
        "f1_macro": f1_score(gt_bin, pred_bin, average="macro", zero_division=0),
        "f1_ir": f1_score(gt_bin, pred_bin, pos_label=1, zero_division=0),
        "precision_ir": precision_score(gt_bin, pred_bin, pos_label=1, zero_division=0),
        "recall_ir": recall_score(gt_bin, pred_bin, pos_label=1, zero_division=0),
    }

    if len(set(gt_bin)) > 1:
        metrics["auc"] = roc_auc_score(gt_bin, pred_bin)
    else:
        metrics["auc"] = None
        print("Warning: only one class in ground truth — AUC undefined.")

    return metrics


def print_metrics(metrics: dict):
    print("\n=== EVALUATION RESULTS ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TSLM on IR prediction")
    parser.add_argument("--model_type", choices=["flamingo", "sp"], required=True)
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pt checkpoint file. Omit for baseline (no fine-tuning).",
    )
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None, help="Cap for smoke-testing")
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument(
        "--labels_path", type=str,
        default="cgm_diabetes/ir_prediction/ir_labels.json",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    label = "pretrained" if args.checkpoint else "baseline"
    print(f"=== IR Evaluation | {args.model_type} | {label} | device={device} ===")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    model = load_model(args.model_type, args.checkpoint, device, args.llm_id)

    dataset = CGMIRDataset(
        split=args.split,
        EOS_TOKEN="",
        labels_path=args.labels_path,
        cache_dir=Path(args.cache_dir),
        max_samples=args.max_samples,
    )
    print(f"Dataset: {len(dataset)} patients ({args.split} split)")

    results = run_evaluation(model, dataset, args.max_new_tokens, args.output)

    metrics = compute_metrics(results)
    print_metrics(metrics)

    metrics_path = args.output.replace(".jsonl", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
