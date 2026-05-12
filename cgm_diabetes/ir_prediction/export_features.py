#
# This source file is part of the TSLM-CGM project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""
Export engineered feature DataFrames for train and test splits to CSV.

Usage (from repo root):
    python -m cgm_diabetes.ir_prediction.export_features

Outputs:
    cgm_diabetes/ir_prediction/data/train_new_feat.csv
    cgm_diabetes/ir_prediction/data/test_new_feat.csv
"""

from pathlib import Path

import pandas as pd

from cgm_diabetes.ir_prediction.data_loader import (
    load_split, _ID_COL, _LABEL_COL, TRAIN_CSV, TEST_CSV,
)

DATA_DIR = Path(__file__).parent / "data"

_CSV = {"train": TRAIN_CSV, "test": TEST_CSV}


def _export(split: str, out_path: Path) -> None:
    print(f"Loading {split} split and computing features...")
    df = load_split(split, include_cgm_features=True)

    # IR_label has been encoded to 0/1 by load_split — save that as IR_label_bool,
    # then restore the original strings from the source CSV.
    orig_labels = pd.read_csv(_CSV[split], usecols=[_ID_COL, _LABEL_COL])
    orig_labels[_ID_COL] = orig_labels[_ID_COL].astype(str)

    df = df.rename(columns={_LABEL_COL: "IR_label_bool"})
    df = df.merge(orig_labels, on=_ID_COL, how="left")

    # Column order: participant_id, IR_label (string), IR_label_bool (0/1), then everything else
    other_cols = [c for c in df.columns if c not in (_ID_COL, _LABEL_COL, "IR_label_bool")]
    df = df[[_ID_COL, _LABEL_COL, "IR_label_bool"] + other_cols]

    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows, {len(df.columns)} columns → {out_path}")


def main() -> None:
    _export("train", DATA_DIR / "train_new_feat.csv")
    print()
    _export("test",  DATA_DIR / "test_new_feat.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
