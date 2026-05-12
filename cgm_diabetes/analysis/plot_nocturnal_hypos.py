#
# This source file is part of the TSLM-CGM project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
# Plot all nocturnal hypoglycemic events (<70 mg/dL, 11pm–7am) across
# patients. Each event is shown with 1 hour of context before the first
# below-range reading and 1 hour after the first reading back in range.
#
# Usage:
#   python -m cgm_diabetes.analysis.plot_nocturnal_hypos
#   python -m cgm_diabetes.analysis.plot_nocturnal_hypos --max_patients 20
#   python -m cgm_diabetes.analysis.plot_nocturnal_hypos --output_dir /tmp/hypos
#

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cgm_diabetes.data.participants import load_participants
from cgm_diabetes.data.cgm_loader import load_cgm_for_patient, DEFAULT_CACHE_DIR

# Clinical thresholds
HYPO_THRESHOLD   = 70   # mg/dL
SEVERE_THRESHOLD = 54   # mg/dL — Level 2

# Nocturnal window
NIGHT_START_HOUR = 23   # 11 pm
NIGHT_END_HOUR   = 7    # 7 am

# Event detection: max gap between consecutive below-70 readings
# to still be counted as one continuous event
MAX_GAP_MINUTES  = 15

# Context padding around each event
CONTEXT_MINUTES  = 60   # 1 hour before first below-range / after first back-in-range


def is_nocturnal(ts: pd.Series) -> pd.Series:
    """Boolean mask for timestamps in the nocturnal window (11pm–7am)."""
    hour = ts.dt.hour
    return (hour >= NIGHT_START_HOUR) | (hour < NIGHT_END_HOUR)


def find_hypo_events(df: pd.DataFrame, threshold: float = HYPO_THRESHOLD) -> list[dict]:
    """
    Find all nocturnal hypoglycemic events in a CGM DataFrame.

    An event is a contiguous run of readings below `threshold` during
    the nocturnal window, with no gap between successive readings exceeding
    MAX_GAP_MINUTES.

    Returns a list of dicts, one per event:
        start_idx  : int   — index of first below-range reading
        end_idx    : int   — index of last below-range reading
        start_time : pd.Timestamp
        end_time   : pd.Timestamp
        nadir      : float — minimum glucose value during event
        nadir_time : pd.Timestamp
        duration_min : float
    """
    ts         = pd.to_datetime(df["timestamp"], utc=True)
    glucose    = df["glucose"].to_numpy()
    night_mask = is_nocturnal(ts).to_numpy()
    below_mask = glucose < threshold

    events = []
    i = 0
    n = len(df)

    while i < n:
        if not (below_mask[i] and night_mask[i]):
            i += 1
            continue

        # Extend event while readings are contiguous and below threshold
        j = i
        while j + 1 < n:
            gap = (ts.iloc[j + 1] - ts.iloc[j]).total_seconds() / 60
            if below_mask[j + 1] and gap <= MAX_GAP_MINUTES:
                j += 1
            else:
                break

        event_glu  = glucose[i:j + 1]
        nadir_rel  = int(np.argmin(event_glu))

        events.append({
            "start_idx":   i,
            "end_idx":     j,
            "start_time":  ts.iloc[i],
            "end_time":    ts.iloc[j],
            "nadir":       round(float(event_glu[nadir_rel]), 1),
            "nadir_time":  ts.iloc[i + nadir_rel],
            "duration_min": round((ts.iloc[j] - ts.iloc[i]).total_seconds() / 60, 1),
        })

        i = j + 1

    return events


def plot_event(
    df: pd.DataFrame,
    event: dict,
    patient_id: str,
    label: str,
    event_num: int,
    total_events: int,
    threshold: float = HYPO_THRESHOLD,
) -> plt.Figure:
    """
    Single-panel plot of one nocturnal hypoglycemic event.

    Window: 1 hour before the first below-range reading to 1 hour after
    the first reading back in range (>=70 mg/dL).
    """
    ts      = pd.to_datetime(df["timestamp"], utc=True)
    glucose = df["glucose"].to_numpy()

    start_time = event["start_time"]
    end_idx    = event["end_idx"]

    # Find first reading back in range after the event
    first_back_in_range = event["end_time"]   # fallback: last below-range reading
    for k in range(end_idx + 1, len(df)):
        gap = (ts.iloc[k] - event["end_time"]).total_seconds() / 60
        if gap > MAX_GAP_MINUTES * 2:
            break
        if glucose[k] >= threshold:
            first_back_in_range = ts.iloc[k]
            break

    plot_start = start_time        - pd.Timedelta(minutes=CONTEXT_MINUTES)
    plot_end   = first_back_in_range + pd.Timedelta(minutes=CONTEXT_MINUTES)

    mask    = (ts >= plot_start) & (ts <= plot_end)
    plot_ts  = ts[mask]
    plot_glu = glucose[mask]

    fig, ax = plt.subplots(figsize=(12, 4))

    fig.suptitle(
        f"Patient {patient_id}  ({label})  —  "
        f"Nocturnal hypo event {event_num}/{total_events}  "
        f"[{start_time.strftime('%Y-%m-%d')}]",
        fontsize=11, fontweight="bold",
    )

    # Shaded hypo zone
    ax.axhspan(0, threshold, color="#ffebee", alpha=0.4, zorder=0)

    # Reference lines — always show both for context, highlight the active one
    ax.axhline(HYPO_THRESHOLD,   color="#2e7d32", linestyle="--", linewidth=1.3,
               label=f"Hypo threshold ({HYPO_THRESHOLD} mg/dL)", zorder=2)
    ax.axhline(SEVERE_THRESHOLD, color="#880000", linestyle=":",  linewidth=1.1,
               label=f"Severe threshold ({SEVERE_THRESHOLD} mg/dL)", zorder=2)

    # Shade the below-range event window
    ax.axvspan(start_time, event["end_time"], color="#ff8f00", alpha=0.18,
               zorder=1, label=f"Below {threshold:.0f} mg/dL window")

    # Draw coloured trace
    _draw_trace(ax, plot_ts, plot_glu)

    # Annotate nadir
    ax.annotate(
        f"Nadir: {event['nadir']:.0f} mg/dL\n{event['nadir_time'].strftime('%H:%M')} UTC",
        xy=(event["nadir_time"], event["nadir"]),
        xytext=(event["nadir_time"], event["nadir"] - 14),
        fontsize=8, color="#880000", ha="center",
        arrowprops=dict(arrowstyle="->", color="#880000", lw=0.9),
    )

    ax.set_ylabel("Glucose (mg/dL)", fontsize=9)
    ax.set_xlabel("Time (UTC)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 15)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    y_min = max(0,   plot_glu.min() - 20) if len(plot_glu) else 0
    y_max = min(350, plot_glu.max() + 30) if len(plot_glu) else 350
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)

    # Duration label in top-left
    ax.text(
        0.01, 0.97,
        f"Duration below {threshold:.0f}: {event['duration_min']:.0f} min",
        transform=ax.transAxes, fontsize=8, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#ccc"),
    )

    fig.tight_layout()
    return fig


def _draw_trace(ax: plt.Axes, ts: pd.Series, glucose: np.ndarray, max_gap_min: float = 10):
    """Draw a CGM trace coloured by zone, skipping sensor dropout gaps."""
    if len(ts) == 0:
        return
    t_arr = ts.to_numpy()
    g_arr = glucose if isinstance(glucose, np.ndarray) else np.asarray(glucose)

    for i in range(len(g_arr) - 1):
        gap = (t_arr[i + 1] - t_arr[i]) / np.timedelta64(1, "m")
        if gap > max_gap_min:
            continue
        mid = (g_arr[i] + g_arr[i + 1]) / 2
        colour = "#1b5e20" if mid < HYPO_THRESHOLD else ("#b71c1c" if mid > 180 else "#1565c0")
        ax.plot([t_arr[i], t_arr[i + 1]], [g_arr[i], g_arr[i + 1]],
                color=colour, linewidth=1.5, zorder=3)

    for t, g in zip(t_arr, g_arr):
        colour = "#1b5e20" if g < HYPO_THRESHOLD else ("#b71c1c" if g > 180 else "#1565c0")
        ax.scatter(t, g, color=colour, s=8, zorder=4)


def run(
    output_dir: Path,
    cache_dir: Optional[Path] = None,
    max_patients: Optional[int] = None,
    split: Optional[str] = None,
    label: Optional[str] = None,
    severe: bool = False,
) -> None:
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    threshold = SEVERE_THRESHOLD if severe else HYPO_THRESHOLD
    if severe:
        print(f"[mode] Severe hypoglycemia mode: threshold = {threshold} mg/dL")

    output_dir.mkdir(parents=True, exist_ok=True)

    participants = load_participants()
    if split is not None:
        participants = {pid: info for pid, info in participants.items()
                        if info["split"] == split}
    if label is not None:
        participants = {pid: info for pid, info in participants.items()
                        if info["label"] == label}
        print(f"[filter] Restricting to label='{label}': {len(participants)} patients")

    patient_ids = list(participants.keys())
    if max_patients is not None:
        patient_ids = patient_ids[:max_patients]

    total_events  = 0
    patients_with = 0

    for pidx, patient_id in enumerate(patient_ids):
        patient_label = participants[patient_id]["label"]

        try:
            df = load_cgm_for_patient(patient_id, cache_dir=cache_dir)
        except Exception as e:
            print(f"  [skip] {patient_id}: {e}")
            continue

        events = find_hypo_events(df, threshold=threshold)

        if not events:
            continue

        patients_with += 1
        print(f"  [{pidx + 1}/{len(patient_ids)}] {patient_id} ({patient_label}): "
              f"{len(events)} nocturnal event(s) below {threshold:.0f} mg/dL")

        patient_dir = output_dir / patient_id
        patient_dir.mkdir(exist_ok=True)

        for eidx, event in enumerate(events):
            total_events += 1

            fig = plot_event(df, event, patient_id, patient_label,
                             event_num=eidx + 1, total_events=len(events),
                             threshold=threshold)

            ts_str = event["start_time"].strftime("%Y%m%d_%H%M")
            fname  = patient_dir / f"event_{eidx + 1:02d}_{ts_str}.png"
            fig.savefig(fname, dpi=110, bbox_inches="tight")
            plt.close(fig)

    print(f"\n{'='*55}")
    print(f"  Threshold            : <{threshold:.0f} mg/dL")
    print(f"  Patients scanned     : {len(patient_ids)}")
    print(f"  Patients with events : {patients_with}")
    print(f"  Total events plotted : {total_events}")
    print(f"  Plots saved to       : {output_dir.resolve()}")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot nocturnal hypoglycemic events (<70 mg/dL, 11pm–7am)."
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("cgm_diabetes/analysis/nocturnal_hypo_plots"),
        help="Directory to write per-patient PNG plots",
    )
    parser.add_argument(
        "--max_patients", type=int, default=None,
        help="Limit to first N patients (default: all)",
    )
    parser.add_argument(
        "--split", type=str, default=None, choices=["train", "val", "test"],
        help="Restrict to one data split (default: all)",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        choices=["healthy", "prediabetes_lifestyle", "oral_non_insulin", "insulin_dependent"],
        help="Restrict to one patient category (default: all)",
    )
    parser.add_argument(
        "--severe", action="store_true",
        help=f"Plot only severe hypoglycemia events (<{SEVERE_THRESHOLD} mg/dL) "
             f"instead of all hypos (<{HYPO_THRESHOLD} mg/dL). "
             f"Default output dir becomes severe_nocturnal_hypo_plots/",
    )
    args = parser.parse_args()

    # If --severe is set and the user didn't explicitly set --output_dir, use a distinct folder
    output_dir = args.output_dir
    if args.severe and output_dir == Path("cgm_diabetes/analysis/nocturnal_hypo_plots"):
        output_dir = Path("cgm_diabetes/analysis/severe_nocturnal_hypo_plots")

    run(
        output_dir=output_dir,
        max_patients=args.max_patients,
        split=args.split,
        label=args.label,
        severe=args.severe,
    )


if __name__ == "__main__":
    main()
