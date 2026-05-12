#
# This source file is part of the TSLM-CGM project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb

_LABEL_COL = "IR_label"
_CV_FOLDS  = 5
_SEED      = 42


def _make_xgb() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=_SEED,
        n_jobs=-1,
        verbosity=0,
    )


def _make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=_SEED,
        n_jobs=-1,
    )


def _build_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     classifier),
    ])


def cross_validate(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str = "xgb",
) -> Dict:
    """
    Run stratified k-fold CV and return mean/std AUC and accuracy.

    Parameters
    ----------
    df          : DataFrame with feature columns and IR_label
    feature_cols: list of column names to use as features
    model_name  : "xgb" or "rf"

    Returns
    -------
    dict with keys: auc_mean, auc_std, acc_mean, acc_std, n_features
    """
    X = df[feature_cols].values
    y = df[_LABEL_COL].values

    clf = _make_xgb() if model_name == "xgb" else _make_rf()
    pipe = _build_pipeline(clf)

    skf   = StratifiedKFold(n_splits=_CV_FOLDS, shuffle=True, random_state=_SEED)
    aucs, accs = [], []

    for train_idx, val_idx in skf.split(X, y):
        pipe.fit(X[train_idx], y[train_idx])
        proba = pipe.predict_proba(X[val_idx])[:, 1]
        preds = pipe.predict(X[val_idx])
        aucs.append(roc_auc_score(y[val_idx], proba))
        accs.append(accuracy_score(y[val_idx], preds))

    return {
        "model":      model_name,
        "n_features": len(feature_cols),
        "auc_mean":   round(float(np.mean(aucs)), 4),
        "auc_std":    round(float(np.std(aucs)),  4),
        "acc_mean":   round(float(np.mean(accs)), 4),
        "acc_std":    round(float(np.std(accs)),  4),
    }


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    feature_cols: List[str],
    model_name: str = "xgb",
) -> Tuple[Pipeline, Dict]:
    """
    Train on the full train split and evaluate on test.

    Returns
    -------
    (fitted_pipeline, metrics_dict)
    """
    X_train = train_df[feature_cols].values
    y_train = train_df[_LABEL_COL].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[_LABEL_COL].values

    clf  = _make_xgb() if model_name == "xgb" else _make_rf()
    pipe = _build_pipeline(clf)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True)

    metrics = {
        "model":      model_name,
        "n_features": len(feature_cols),
        "test_auc":   round(float(roc_auc_score(y_test, proba)), 4),
        "test_acc":   round(float(accuracy_score(y_test, preds)), 4),
        "precision_ir1": round(report.get("1", {}).get("precision", np.nan), 4),
        "recall_ir1":    round(report.get("1", {}).get("recall", np.nan),    4),
        "f1_ir1":        round(report.get("1", {}).get("f1-score", np.nan),  4),
    }
    return pipe, metrics


def print_feature_importances(
    pipe: Pipeline,
    feature_cols: List[str],
    top_n: int = 20,
) -> None:
    """Print top-N feature importances from a fitted XGB or RF pipeline."""
    clf = pipe.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        print("  [warning] model has no feature_importances_ attribute")
        return

    idx  = np.argsort(importances)[::-1]
    print(f"\n  Top {min(top_n, len(feature_cols))} features:")
    for rank, i in enumerate(idx[:top_n], start=1):
        print(f"    {rank:>3}. {feature_cols[i]:<45}  {importances[i]:.4f}")
