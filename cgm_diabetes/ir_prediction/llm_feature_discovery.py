#
# This source file is part of the TSLM-CGM project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""
Prints a structured prompt to the console for LLM-assisted feature discovery,
and optionally generates comparison plots for image-based LLM analysis.

Usage:
    python -m cgm_diabetes.ir_prediction.llm_feature_discovery           # v2 text prompt
    python -m cgm_diabetes.ir_prediction.llm_feature_discovery --v1      # original prompt
    python -m cgm_diabetes.ir_prediction.llm_feature_discovery --plot    # generate plots only
    python -m cgm_diabetes.ir_prediction.llm_feature_discovery --plot --v2  # plots + text prompt
"""

import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd

from cgm_diabetes.data.cgm_loader import load_cgm_for_patient, DEFAULT_CACHE_DIR
from cgm_diabetes.ir_prediction.data_loader import (
    load_split, get_feature_columns, _LABEL_COL, _ID_COL,
)

_DIVIDER = "=" * 72
_SEP     = "-" * 72

# ── Importances from best model (base_plus_greedy_cgm RF, AUC 0.9123) ─────────
BEST_MODEL_IMPORTANCES = [
    ("bmi_vsorres..BMI",                          0.3062),
    ("whr_vsorres..Waist.to.Hip.Ratio..WHR.",     0.1283),
    ("wake_to_first_mean_fasting_glucose_daily",  0.1021),
    ("wake_to_first_sd_fasting_glucose_daily",    0.0873),
    ("cgm_hyper_episode_duration_mean",           0.0772),
    ("cgm_auc_above_180",                         0.0666),
    ("age",                                       0.0657),
    ("sd_hr_day_night_diff",                      0.0578),
    ("overall_daily_tir_70_180_pct",              0.0533),
    ("stress_overall_daily_mean_stress",          0.0453),
    ("cgm_n_hypo_l2_events",                      0.0102),
]

# ── One-at-a-time sweep results (delta vs base-only RF AUC 0.8722) ────────────
SINGLE_FEATURE_DELTAS = {
    "cgm_hyper_episode_duration_mean":  +0.0113,
    "cgm_postprandial_auc_2h":          -0.0050,
    "cgm_hyper_duration_per_bmi":       -0.0050,
    "cgm_excursion_auc_normalized":     -0.0113,
    "cgm_rise_fall_ratio":              -0.0151,
    "cgm_hyper_recovery_slope":         -0.0151,
    "cgm_fasting_glucose_trend_slope":  -0.0151,
    "cgm_dawn_phenomenon_magnitude":    -0.0188,
    "cgm_n_hypo_l2_events":             -0.0213,
    "cgm_lbgi":                         -0.0213,
    "cgm_glucose_entropy":              -0.0263,
    "cgm_autocorr_lag5":                -0.0251,
    "cgm_interday_variability":         -0.0276,
    "cgm_postprandial_peak_time_mean":  -0.0288,
    "cgm_overnight_slope_mean":         -0.0226,
    "cgm_circadian_amplitude":          -0.0313,
    "cgm_roc_positive_negative_ratio":  -0.0326,
    "cgm_fasting_cv":                   -0.0338,
    "cgm_glucose_range_night_vs_day":   -0.0213,
    "cgm_meal_count_estimated":         -0.0351,
}


def _build_24h_profile(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each 30-minute slot of the day (48 slots), compute mean glucose
    separately for IR=0 and IR=1 patients using cached CGM traces.
    Returns a DataFrame with columns [slot_label, ir0_mean, ir1_mean, delta].
    """
    slots    = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
    ir0_vals = {s: [] for s in slots}
    ir1_vals = {s: [] for s in slots}

    for _, row in train_df.iterrows():
        pid   = str(row[_ID_COL])
        label = int(row[_LABEL_COL])
        try:
            cgm = load_cgm_for_patient(pid, cache_dir=DEFAULT_CACHE_DIR)
        except Exception:
            continue

        cgm = cgm.copy()
        cgm["hour"]   = pd.to_datetime(cgm["timestamp"], utc=True).dt.hour
        cgm["minute"] = pd.to_datetime(cgm["timestamp"], utc=True).dt.minute
        cgm["slot"]   = cgm.apply(
            lambda r: f"{r['hour']:02d}:{'00' if r['minute'] < 30 else '30'}", axis=1
        )

        patient_means = cgm.groupby("slot")["glucose"].mean()
        target = ir0_vals if label == 0 else ir1_vals
        for s in slots:
            if s in patient_means.index:
                target[s].append(patient_means[s])

    rows = []
    for s in slots:
        m0 = float(np.mean(ir0_vals[s])) if ir0_vals[s] else np.nan
        m1 = float(np.mean(ir1_vals[s])) if ir1_vals[s] else np.nan
        rows.append({"time": s, "ir0_mean": round(m0, 1), "ir1_mean": round(m1, 1),
                     "delta": round(m1 - m0, 1)})
    return pd.DataFrame(rows)


def print_discovery_prompt_v2(train_df: pd.DataFrame) -> None:
    """
    Print a second-iteration feature discovery prompt that includes:
    - All experimental results to date
    - Stratified 24h CGM glucose profiles (IR=0 vs IR=1)
    - Class-conditional statistics for all computed features
    - Targeted questions about what signal might be missing
    """
    base_cols = get_feature_columns(train_df, "base")
    cgm_cols  = get_feature_columns(train_df, "cgm")
    label     = train_df[_LABEL_COL]
    n0, n1    = int((label == 0).sum()), int((label == 1).sum())

    print(_DIVIDER)
    print("  LLM FEATURE DISCOVERY PROMPT — ITERATION 2")
    print("  Copy everything between the outer dividers into Claude.")
    print(_DIVIDER)
    print()

    # ── Task & dataset ────────────────────────────────────────────────────────
    print("## Task")
    print(textwrap.fill(
        "Binary classification: predict Insulin Resistance (IR_label = 1 vs 0) "
        "in patients wearing a Dexcom G6 CGM sensor (5-minute readings, glucose "
        "in mg/dL). Each row is one patient. The dataset is small: 97 train, "
        "61 test patients.",
        width=72,
    ))
    print(f"\n  IR=1 (insulin resistant): {n1}  ({100*n1/(n0+n1):.1f}%)")
    print(f"  IR=0 (not IR):            {n0}  ({100*n0/(n0+n1):.1f}%)")
    print()

    # ── Best model ────────────────────────────────────────────────────────────
    print("## Best model so far")
    print("  Random Forest, 11 features = 8 base CSV + 3 CGM-derived")
    print("  Test AUC = 0.9123  |  Accuracy = 0.836  |  F1(IR=1) = 0.737")
    print("  (Note: 3 CGM features were selected by greedy test-set search —")
    print("   unbiased CV estimate is base-only AUC 0.8722; CGM gain uncertain.)")
    print()
    print("  Feature importances (RF, best model):")
    for feat, imp in BEST_MODEL_IMPORTANCES:
        tag = "  [CGM]" if feat.startswith("cgm_") else "  [BASE]"
        print(f"    {feat:<45}  {imp:.4f}{tag}")
    print()

    # ── What was tried ────────────────────────────────────────────────────────
    print("## All CGM features computed and tested (60 total)")
    print("  Only ONE feature improves AUC when added alone to the base model:")
    print("  cgm_hyper_episode_duration_mean  (+0.011)")
    print()
    print("  Every other CGM feature HURTS when added to the base model alone.")
    print("  Δ vs base-only RF AUC 0.8722 (top 20 shown):")
    for feat, delta in sorted(SINGLE_FEATURE_DELTAS.items(), key=lambda x: -x[1])[:20]:
        marker = " ◄ HELPS" if delta > 0 else ""
        print(f"    {feat:<45}  {delta:>+.4f}{marker}")
    print()
    print("  Features already computed (full list — do NOT re-suggest these):")
    for col in sorted(cgm_cols):
        print(f"    {col}")
    print()

    # ── 24h stratified CGM profile ────────────────────────────────────────────
    print("## Stratified 24-hour glucose profiles (averaged across all days per patient,")
    print("   then averaged by IR class across patients)")
    print("  Computing from cached CGM traces...")
    profile = _build_24h_profile(train_df)
    print()
    print(f"  {'Time':<6}  {'IR=0 mean':>9}  {'IR=1 mean':>9}  {'Δ (IR1−IR0)':>11}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*11}")
    for _, row in profile.iterrows():
        bar  = "▲" if row["delta"] > 5 else ("▼" if row["delta"] < -5 else " ")
        print(f"  {row['time']:<6}  {row['ir0_mean']:>9.1f}  {row['ir1_mean']:>9.1f}  "
              f"{row['delta']:>+11.1f} {bar}")
    print()

    # ── Class-conditional feature statistics ──────────────────────────────────
    print("## Class-conditional statistics for all CGM features")
    print(f"  {'Feature':<45}  {'IR=0':>7}  {'IR=1':>7}  {'Δ':>7}  {'|Δ|/SD':>7}")
    print(f"  {'-'*45}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    rows = []
    for col in cgm_cols:
        col_data = train_df[col].dropna()
        if len(col_data) < 10:
            continue
        m0   = train_df.loc[label == 0, col].mean()
        m1   = train_df.loc[label == 1, col].mean()
        sd   = col_data.std()
        effect = abs(m1 - m0) / sd if sd > 0 else 0
        rows.append((col, m0, m1, m1 - m0, effect))
    for col, m0, m1, delta, effect in sorted(rows, key=lambda x: -x[4]):
        print(f"  {col:<45}  {m0:>7.2f}  {m1:>7.2f}  {delta:>+7.2f}  {effect:>7.3f}")
    print()

    # ── Targeted questions ────────────────────────────────────────────────────
    print("## Your task — suggest NEW CGM features")
    print()
    print(textwrap.fill(
        "Given everything above — including the 24h glucose profiles, the "
        "class-conditional feature statistics, and the full list of features "
        "already tried — what ADDITIONAL features computed from the raw CGM "
        "time series might help predict insulin resistance?",
        width=72,
    ))
    print()
    print("Key constraints:")
    print("  - Dataset is very small (97 train). Features must be robust to noise.")
    print("  - Base features (BMI, WHR, fasting glucose) already explain most variance.")
    print("  - The winning CGM features capture slow glucose CLEARANCE (duration +")
    print("    area of hyperglycemic episodes). New features should be orthogonal")
    print("    to these.")
    print("  - CV-wrapped feature selection found NO stable CGM features — any")
    print("    new feature needs to be biologically compelling, not just correlated.")
    print()
    print("Please look at the 24h profiles above and tell me:")
    print("  1. At which times of day do IR=0 and IR=1 patients diverge most?")
    print("  2. What shape/pattern differences do you see in the profiles?")
    print("  3. Suggest 5-8 NEW features based on those patterns. For each:")
    print("     a) Feature name (snake_case, prefix cgm_)")
    print("     b) Exact computation from the raw 5-min CGM series")
    print("     c) Why it should be orthogonal to features already tried")
    print("     d) Biological mechanism linking it to IR")
    print()
    print(_DIVIDER)
    print("  END OF PROMPT")
    print(_DIVIDER)


def print_discovery_prompt_v1(train_df: pd.DataFrame) -> None:
    """Original first-iteration prompt (column names + basic stats only)."""
    base_cols = get_feature_columns(train_df, "base")
    cgm_cols  = get_feature_columns(train_df, "cgm")
    label     = train_df[_LABEL_COL]
    pos, neg  = int(label.sum()), len(label) - int(label.sum())

    print(_DIVIDER)
    print("  LLM FEATURE DISCOVERY PROMPT — ITERATION 1 (original)")
    print(_DIVIDER)
    print()
    print(f"IR=1: {pos}, IR=0: {neg}")
    print()
    for col in base_cols:
        stats = train_df[col].describe()
        print(f"  {col}  mean={stats['mean']:.2f}  std={stats['std']:.2f}")
    print()
    if cgm_cols:
        print("CGM features already computed:")
        for col in cgm_cols:
            if not train_df[col].isna().all():
                stats = train_df[col].dropna().describe()
                print(f"  {col:<45}  mean={stats['mean']:.2f}  std={stats['std']:.2f}")
    print(_DIVIDER)


_PLOTS_DIR   = Path(__file__).parent / "plots"
_COLORS      = {"ir0": "#4878CF", "ir1": "#E84646"}   # blue / red
_HYPO_COLOR  = "#FFF0A0"
_HYPER_COLOR = "#FFD0D0"
_N_SAMPLES   = 6   # patients per class in the sample-traces plot


def _build_per_patient_24h(train_df: pd.DataFrame) -> dict:
    """Return {pid: array(48,)} of mean glucose per 30-min slot, per patient."""
    profiles = {}
    for _, row in train_df.iterrows():
        pid = str(row[_ID_COL])
        try:
            cgm = load_cgm_for_patient(pid, cache_dir=DEFAULT_CACHE_DIR)
        except Exception:
            continue
        ts   = pd.to_datetime(cgm["timestamp"], utc=True)
        slot = (ts.dt.hour * 2 + (ts.dt.minute >= 30).astype(int)).values
        means = np.full(48, np.nan)
        for s in range(48):
            vals = cgm["glucose"].values[slot == s]
            if len(vals) > 0:
                means[s] = vals.mean()
        profiles[pid] = means
    return profiles


def _plot_mean_profile(train_df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Plot 1 — Mean ± 1 SD 24h glucose profile for IR=0 vs IR=1,
    with a difference panel below.
    """
    profiles = _build_per_patient_24h(train_df)
    label    = train_df.set_index(_ID_COL)[_LABEL_COL]

    ir0_mat = np.array([profiles[p] for p in profiles if label.get(p, -1) == 0])
    ir1_mat = np.array([profiles[p] for p in profiles if label.get(p, -1) == 1])

    times  = np.arange(48) * 0.5          # fractional hours 0–23.5
    xticks = np.arange(0, 24, 3)
    xlabels= [f"{h:02d}:00" for h in xticks]

    fig, (ax_main, ax_diff) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.suptitle("Mean 24h Glucose Profile — IR=0 vs IR=1 (train set)",
                 fontsize=14, fontweight="bold", y=0.98)

    for mat, key, label_str in [
        (ir0_mat, "ir0", f"Non-IR  (n={len(ir0_mat)})"),
        (ir1_mat, "ir1", f"Insulin Resistant  (n={len(ir1_mat)})"),
    ]:
        col  = _COLORS[key]
        mean = np.nanmean(mat, axis=0)
        sd   = np.nanstd(mat, axis=0)
        ax_main.plot(times, mean, color=col, lw=2.5, label=label_str)
        ax_main.fill_between(times, mean - sd, mean + sd, color=col, alpha=0.18)

    # Horizontal reference bands
    ax_main.axhspan(70, 180, color="#E8F5E9", alpha=0.4, zorder=0)
    ax_main.axhline(70,  color="#66BB6A", lw=0.8, ls="--", alpha=0.7)
    ax_main.axhline(180, color="#EF5350", lw=0.8, ls="--", alpha=0.7)
    ax_main.set_ylabel("Glucose (mg/dL)", fontsize=11)
    ax_main.set_ylim(60, 220)
    ax_main.legend(fontsize=11, loc="upper left")
    ax_main.grid(axis="y", alpha=0.3)

    # Difference panel
    m0    = np.nanmean(ir0_mat, axis=0)
    m1    = np.nanmean(ir1_mat, axis=0)
    delta = m1 - m0
    colors = [_COLORS["ir1"] if d > 0 else _COLORS["ir0"] for d in delta]
    ax_diff.bar(times, delta, width=0.45, color=colors, alpha=0.75)
    ax_diff.axhline(0, color="black", lw=0.8)
    ax_diff.set_ylabel("Δ IR1−IR0\n(mg/dL)", fontsize=10)
    ax_diff.set_xlabel("Time of day", fontsize=11)
    ax_diff.set_xticks(xticks)
    ax_diff.set_xticklabels(xlabels)
    ax_diff.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = out_dir / "plot1_mean_24h_profile.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_patient_heatmap(train_df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Plot 2 — Heatmap of every patient's mean 24h glucose profile,
    sorted IR=0 first then IR=1. Rows = patients, columns = 30-min slots.
    """
    profiles = _build_per_patient_24h(train_df)
    label    = train_df.set_index(_ID_COL)[_LABEL_COL]

    ir0_pids = [p for p in profiles if label.get(p, -1) == 0]
    ir1_pids = [p for p in profiles if label.get(p, -1) == 1]
    ordered  = ir0_pids + ir1_pids

    matrix   = np.array([profiles[p] for p in ordered])
    n0, n1   = len(ir0_pids), len(ir1_pids)

    xticks  = list(range(0, 48, 6))
    xlabels = [f"{h:02d}:00" for h in range(0, 24, 3)]

    fig, ax = plt.subplots(figsize=(14, max(8, len(ordered) * 0.22)))
    fig.suptitle("Per-Patient 24h Glucose Heatmap — sorted by IR class (train set)",
                 fontsize=13, fontweight="bold")

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r",
                   vmin=70, vmax=220, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Glucose (mg/dL)", shrink=0.6)

    # Class boundary
    ax.axhline(n0 - 0.5, color="black", lw=2.5)
    ax.text(48.3, n0 / 2, f"Non-IR\n(n={n0})", va="center", fontsize=9,
            color=_COLORS["ir0"], fontweight="bold")
    ax.text(48.3, n0 + n1 / 2, f"IR\n(n={n1})", va="center", fontsize=9,
            color=_COLORS["ir1"], fontweight="bold")

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Time of day", fontsize=11)
    ax.set_ylabel("Patient (sorted by class)", fontsize=11)
    ax.set_yticks([])

    plt.tight_layout()
    out = out_dir / "plot2_patient_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_sample_traces(train_df: pd.DataFrame, out_dir: Path,
                        n_samples: int = _N_SAMPLES) -> Path:
    """
    Plot 3 — Grid of continuous multi-day CGM traces for representative
    patients from each class. Left column = IR=0, right = IR=1.
    """
    label    = train_df.set_index(_ID_COL)[_LABEL_COL]
    ir0_pids = label[label == 0].index.tolist()[:n_samples]
    ir1_pids = label[label == 1].index.tolist()[:n_samples]

    n_rows = max(len(ir0_pids), len(ir1_pids))
    fig    = plt.figure(figsize=(16, n_rows * 2.2))
    fig.suptitle(f"Sample Individual CGM Traces (first {n_samples} patients per class)",
                 fontsize=13, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(n_rows, 2, hspace=0.5, wspace=0.25)

    def _draw_trace(ax, pid, ir_class):
        try:
            cgm = load_cgm_for_patient(pid, cache_dir=DEFAULT_CACHE_DIR)
        except Exception:
            ax.text(0.5, 0.5, f"{pid}\n(no data)", ha="center", va="center",
                    transform=ax.transAxes)
            return
        ts = pd.to_datetime(cgm["timestamp"], utc=True)
        t  = (ts - ts.iloc[0]).dt.total_seconds() / 86400   # days from start
        g  = cgm["glucose"].values
        col = _COLORS["ir1"] if ir_class == 1 else _COLORS["ir0"]
        ax.axhspan(70, 180, color="#E8F5E9", alpha=0.35, zorder=0)
        ax.axhline(70,  color="#66BB6A", lw=0.6, ls="--", alpha=0.6)
        ax.axhline(180, color="#EF5350", lw=0.6, ls="--", alpha=0.6)
        ax.plot(t, g, lw=0.7, color=col, alpha=0.85)
        ax.set_ylim(40, 280)
        label_str = "IR" if ir_class == 1 else "Non-IR"
        ax.set_title(f"Patient {pid}  [{label_str}]", fontsize=9, color=col)
        ax.set_ylabel("mg/dL", fontsize=7)
        ax.set_xlabel("Days", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.2)

    for i, pid in enumerate(ir0_pids):
        _draw_trace(fig.add_subplot(gs[i, 0]), pid, ir_class=0)
    for i, pid in enumerate(ir1_pids):
        _draw_trace(fig.add_subplot(gs[i, 1]), pid, ir_class=1)

    # Column headers
    fig.text(0.26, 1.0, "Non-IR (IR=0)", ha="center", fontsize=12,
             fontweight="bold", color=_COLORS["ir0"])
    fig.text(0.74, 1.0, "Insulin Resistant (IR=1)", ha="center", fontsize=12,
             fontweight="bold", color=_COLORS["ir1"])

    plt.tight_layout()
    out = out_dir / "plot3_sample_traces.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_comparison_plots(train_df: pd.DataFrame) -> None:
    """Generate all three comparison plots and save to the plots/ directory."""
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[plots] Building per-patient 24h profiles from CGM cache...")
    print("[plots] Generating plot 1 — mean 24h profile comparison...")
    p1 = _plot_mean_profile(train_df, _PLOTS_DIR)
    print(f"  Saved: {p1}")

    print("[plots] Generating plot 2 — patient heatmap...")
    p2 = _plot_patient_heatmap(train_df, _PLOTS_DIR)
    print(f"  Saved: {p2}")

    print("[plots] Generating plot 3 — sample individual traces...")
    p3 = _plot_sample_traces(train_df, _PLOTS_DIR)
    print(f"  Saved: {p3}")

    print(f"\n[plots] All plots saved to: {_PLOTS_DIR}/")
    print("  Drag and drop any or all of these images into your Claude")
    print("  conversation and ask for feature suggestions based on the patterns.")


if __name__ == "__main__":
    do_plot = "--plot" in sys.argv
    do_v2   = "--v2"   in sys.argv
    do_v1   = "--v1"   in sys.argv

    # Plots only need the base CSV (no CGM feature computation required)
    need_cgm = not do_plot or do_v2
    print(f"[llm_feature_discovery] Loading train split "
          f"(include_cgm_features={need_cgm})...")
    train_df = load_split("train", include_cgm_features=need_cgm)

    if do_plot:
        generate_comparison_plots(train_df)
    if do_v2:
        print_discovery_prompt_v2(train_df)
    if do_v1:
        print_discovery_prompt_v1(train_df)
    if not do_plot and not do_v1 and not do_v2:
        # Default: v2 text prompt
        print_discovery_prompt_v2(train_df)
