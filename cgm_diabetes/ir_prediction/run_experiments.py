#
# This source file is part of the TSLM-CGM project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""
Run iterative IR-prediction experiments across different feature sets and models.

Usage (from repo root):
    python -m cgm_diabetes.ir_prediction.run_experiments

After running llm_feature_discovery.py and getting suggestions,
add the suggested feature column names to CUSTOM_FEATURES below, then re-run.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

from cgm_diabetes.ir_prediction.data_loader import load_split, get_feature_columns, _LABEL_COL
from cgm_diabetes.ir_prediction.models import (
    cross_validate,
    train_and_evaluate,
    print_feature_importances,
)

# Set to a non-empty list to enable the 'custom' feature set for ad-hoc experiments.
CUSTOM_FEATURES: list = []

# Number of top CGM features to select (by univariate F-test) for the
# base_plus_cgm_selected experiment.
TOP_K_CGM = 8

# Maximum steps in the CV-wrapped greedy forward selection (train-only, unbiased).
GREEDY_MAX_STEPS = 5

_DIVIDER = "=" * 72
_MODELS  = ["xgb", "rf"]
_SEED    = 42


def _select_top_cgm_features(train_df: pd.DataFrame, cgm_cols: List[str], k: int) -> List[str]:
    """
    Return the top-k CGM features ranked by univariate F-test on the train set.
    Imputes NaNs with column medians before scoring.
    """
    X = SimpleImputer(strategy="median").fit_transform(train_df[cgm_cols].values)
    y = train_df[_LABEL_COL].values
    k = min(k, len(cgm_cols))
    selector = SelectKBest(f_classif, k=k).fit(X, y)
    selected = [cgm_cols[i] for i in selector.get_support(indices=True)]
    scores   = dict(zip(cgm_cols, selector.scores_))
    print(f"\n  Top-{k} CGM features by F-score:")
    for feat in sorted(selected, key=lambda c: scores[c], reverse=True):
        print(f"    {feat:<45}  F={scores[feat]:.2f}")
    return selected


def _print_cv_result(result: dict) -> None:
    print(
        f"  {result['model'].upper():<5}  "
        f"AUC {result['auc_mean']:.4f} ± {result['auc_std']:.4f}  |  "
        f"Acc {result['acc_mean']:.4f} ± {result['acc_std']:.4f}  "
        f"(n_features={result['n_features']})"
    )


def _print_test_result(result: dict) -> None:
    print(
        f"  {result['model'].upper():<5}  "
        f"AUC={result['test_auc']:.4f}  Acc={result['test_acc']:.4f}  "
        f"P={result['precision_ir1']:.4f}  R={result['recall_ir1']:.4f}  "
        f"F1={result['f1_ir1']:.4f}"
    )


def _one_auc(train_df: pd.DataFrame, test_df: pd.DataFrame, feat_cols: List[str]) -> float:
    """Train an RF on feat_cols and return test AUC. Used for the addition sweep."""
    _, result = train_and_evaluate(train_df, test_df, feat_cols, model_name="rf")
    return result["test_auc"]


def cv_greedy_selection(
    train_df: pd.DataFrame,
    base_cols: List[str],
    cgm_cols: List[str],
    n_folds: int = 5,
    max_steps: int = 5,
    min_folds: int = 3,
) -> List[str]:
    """
    Run greedy forward CGM feature selection inside CV folds on the train set.

    In each fold the greedy search uses only that fold's val split to score
    candidate features — the test set is never seen. Features selected in
    >= min_folds folds are returned as the consensus set.

    Parameters
    ----------
    min_folds : a feature must appear in at least this many folds (out of n_folds)
                to be included in the final set. Default 3 = simple majority.
    """
    from sklearn.model_selection import StratifiedKFold

    y   = train_df[_LABEL_COL].values
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=_SEED)
    selection_counts: dict = {}

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_train = train_df.iloc[tr_idx].reset_index(drop=True)
        fold_val   = train_df.iloc[val_idx].reset_index(drop=True)

        current_set = list(base_cols)
        current_auc = _one_auc(fold_train, fold_val, current_set)
        remaining   = list(cgm_cols)
        selected    = []

        for _ in range(max_steps):
            best_feat, best_auc = None, current_auc
            for feat in remaining:
                auc = _one_auc(fold_train, fold_val, current_set + [feat])
                if auc > best_auc:
                    best_auc, best_feat = auc, feat
            if best_feat is None:
                break
            current_set.append(best_feat)
            selected.append(best_feat)
            remaining.remove(best_feat)
            current_auc = best_auc

        print(f"  Fold {fold_idx + 1}: {selected if selected else '(none)'}")
        for feat in selected:
            selection_counts[feat] = selection_counts.get(feat, 0) + 1

    print(f"\n  Feature selection counts across {n_folds} folds (need ≥{min_folds}):")
    for feat, count in sorted(selection_counts.items(), key=lambda x: -x[1]):
        marker = " ◄" if count >= min_folds else ""
        print(f"    {feat:<45}  {count}/{n_folds}{marker}")

    return [f for f, c in sorted(selection_counts.items(), key=lambda x: -x[1])
            if c >= min_folds]


def run_all() -> None:
    # ── Load data ──────────────────────────────────────────────────────────────
    print(_DIVIDER)
    print("Loading train split...")
    train_df = load_split("train", include_cgm_features=True)

    print("Loading test split...")
    test_df  = load_split("test",  include_cgm_features=True)
    print(_DIVIDER)

    # ── Define feature sets ────────────────────────────────────────────────────
    base_cols = get_feature_columns(train_df, "base")
    cgm_cols  = get_feature_columns(train_df, "cgm")

    print(f"\nSelecting top-{TOP_K_CGM} CGM features from {len(cgm_cols)} candidates...")
    top_cgm = _select_top_cgm_features(train_df, cgm_cols, k=TOP_K_CGM)

    feature_sets = {
        "base":                   base_cols,
        "base_plus_cgm_selected": base_cols + top_cgm,
        "cgm_only":               cgm_cols,
        "all":                    get_feature_columns(train_df, "all"),
    }

    if CUSTOM_FEATURES:
        valid = [f for f in CUSTOM_FEATURES if f in train_df.columns]
        missing = set(CUSTOM_FEATURES) - set(valid)
        if missing:
            print(f"[warning] Custom features not found in data: {missing}")
        feature_sets["custom"] = valid

    # ── Cross-validation on train ──────────────────────────────────────────────
    print("\nCROSS-VALIDATION RESULTS (train set, 5-fold stratified)")
    print(_DIVIDER)
    cv_summary = []

    for fs_name, feat_cols in feature_sets.items():
        if not feat_cols:
            print(f"\n  [{fs_name}] — no features, skipping")
            continue
        print(f"\n  Feature set: {fs_name!r}  ({len(feat_cols)} features)")
        for model_name in _MODELS:
            result = cross_validate(train_df, feat_cols, model_name=model_name)
            _print_cv_result(result)
            cv_summary.append({"feature_set": fs_name, **result})

    # ── Test-set evaluation ────────────────────────────────────────────────────
    print("\n\nTEST SET RESULTS")
    print(_DIVIDER)
    test_summary = []
    best_pipes   = {}

    for fs_name, feat_cols in feature_sets.items():
        if not feat_cols:
            continue
        print(f"\n  Feature set: {fs_name!r}  ({len(feat_cols)} features)")
        for model_name in _MODELS:
            pipe, result = train_and_evaluate(train_df, test_df, feat_cols, model_name)
            _print_test_result(result)
            best_pipes[f"{fs_name}_{model_name}"] = (pipe, feat_cols)
            test_summary.append({"feature_set": fs_name, **result})

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n\nSUMMARY TABLE (test AUC)")
    print(_DIVIDER)
    summary_df = pd.DataFrame(test_summary)
    pivot = summary_df.pivot(index="feature_set", columns="model", values="test_auc")
    print(pivot.to_string())

    # ── Feature importances for best model ────────────────────────────────────
    best_row = summary_df.loc[summary_df["test_auc"].idxmax()]
    best_key = f"{best_row['feature_set']}_{best_row['model']}"
    print(f"\n\nFEATURE IMPORTANCES — best model: {best_key} (AUC={best_row['test_auc']:.4f})")
    print(_DIVIDER)
    pipe, feat_cols = best_pipes[best_key]
    print_feature_importances(pipe, feat_cols, top_n=25)

    # ── One-at-a-time CGM addition sweep (exploratory, not used for selection) ──
    print("\n\nONE-AT-A-TIME CGM ADDITION SWEEP (base + 1 CGM feature, RF, test AUC)")
    print("  NOTE: exploratory ranking only — these rankings must not be used to select")
    print("  a multi-feature set and re-evaluate on the same test set.")
    print(_DIVIDER)
    baseline_auc = _one_auc(train_df, test_df, base_cols)
    print(f"  Baseline (base only):  RF AUC = {baseline_auc:.4f}")
    print(f"  {'Feature':<45}  {'test AUC':>8}  {'Δ vs base':>10}")
    print(f"  {'-'*45}  {'-'*8}  {'-'*10}")
    sweep_results = []
    for feat in cgm_cols:
        auc = _one_auc(train_df, test_df, base_cols + [feat])
        sweep_results.append((feat, auc, auc - baseline_auc))
    sweep_results.sort(key=lambda x: x[1], reverse=True)
    for feat, auc, delta in sweep_results[:20]:
        marker = " ◄" if delta > 0 else ""
        print(f"  {feat:<45}  {auc:>8.4f}  {delta:>+10.4f}{marker}")

    # ── CV-wrapped greedy selection (unbiased) ────────────────────────────────
    print("\n\nCV-WRAPPED GREEDY SELECTION (test set never seen during selection)")
    print(_DIVIDER)
    print("  Running 5-fold CV greedy search on train set only...")
    cv_cgm = cv_greedy_selection(train_df, base_cols, cgm_cols,
                                 n_folds=5, max_steps=GREEDY_MAX_STEPS, min_folds=3)

    print(f"\n  CV-consensus CGM features: {cv_cgm if cv_cgm else '(none reached majority)'}")
    if cv_cgm:
        cv_feat_cols = base_cols + cv_cgm
        cv_pipe, cv_result = train_and_evaluate(train_df, test_df, cv_feat_cols, "rf")
        print(f"  Train-all → test:  AUC={cv_result['test_auc']:.4f}  "
              f"Acc={cv_result['test_acc']:.4f}  "
              f"P={cv_result['precision_ir1']:.4f}  R={cv_result['recall_ir1']:.4f}  "
              f"F1={cv_result['f1_ir1']:.4f}")
        print(f"  Δ vs base-only:    {cv_result['test_auc'] - baseline_auc:+.4f}")
        print(f"\n  Feature importances for CV-selected model:")
        print_feature_importances(cv_pipe, cv_feat_cols, top_n=15)

    print("\n[done]")


if __name__ == "__main__":
    run_all()
