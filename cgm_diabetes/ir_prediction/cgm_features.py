#
# This source file is part of the TSLM-CGM project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

import numpy as np
import pandas as pd
from typing import Dict


# Threshold constants (mg/dL)
_HYPO_L2  = 54
_HYPO_L1  = 70
_TARGET_H = 180
_HYPER_L2 = 250

# Circadian split: day = 06:00–22:00
_DAY_START = 6
_DAY_END   = 22

# Minimum continuous minutes to count as an event (at 5-min sampling = 3 readings)
_EVENT_MIN_READINGS = 3


def _require_columns(df: pd.DataFrame) -> None:
    for col in ("timestamp", "glucose"):
        if col not in df.columns:
            raise ValueError(f"CGM DataFrame must have a '{col}' column")


def _tir(glucose: pd.Series, lo: float, hi: float) -> float:
    return float(glucose.between(lo, hi).mean() * 100)


def _mage(glucose: pd.Series) -> float:
    """Mean Amplitude of Glycemic Excursions (simplified: mean of peak-to-nadir swings > 1 SD)."""
    arr = glucose.values
    sd  = arr.std()
    if sd == 0:
        return 0.0
    peaks  = []
    nadirs = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            peaks.append(arr[i])
        elif arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            nadirs.append(arr[i])
    if not peaks or not nadirs:
        return 0.0
    excursions = []
    for p in peaks:
        for n in nadirs:
            diff = abs(p - n)
            if diff > sd:
                excursions.append(diff)
    return float(np.mean(excursions)) if excursions else 0.0


def _count_events(condition: pd.Series) -> int:
    """Count distinct runs of True in `condition` that last >= _EVENT_MIN_READINGS."""
    runs  = (condition != condition.shift()).cumsum()
    count = 0
    for _, group in condition.groupby(runs):
        if group.iloc[0] and len(group) >= _EVENT_MIN_READINGS:
            count += 1
    return count


def _auc_above(glucose: pd.Series, threshold: float, interval_min: float = 5.0) -> float:
    """Trapezoidal AUC above threshold (mg/dL·min)."""
    excess = (glucose - threshold).clip(lower=0)
    # np.trapz was removed in NumPy 2.0; np.trapezoid is the replacement
    trapz = getattr(np, "trapezoid", np.trapezoid)
    return float(trapz(excess, dx=interval_min))


def _rate_of_change(glucose: pd.Series) -> pd.Series:
    return glucose.diff()


def _sample_entropy(arr: np.ndarray, m: int = 2, r_factor: float = 0.2, max_n: int = 1000) -> float:
    """Sample entropy (SampEn) — lower values indicate reduced glycemic complexity."""
    if len(arr) < m + 2:
        return np.nan
    r = r_factor * np.std(arr, ddof=1)
    if r == 0:
        return np.nan
    if len(arr) > max_n:
        arr = arr[np.linspace(0, len(arr) - 1, max_n, dtype=int)]
    N = len(arr)
    templates_m  = np.lib.stride_tricks.sliding_window_view(arr, m)
    templates_m1 = np.lib.stride_tricks.sliding_window_view(arr, m + 1)
    n = len(templates_m1)
    A = B = 0
    for i in range(n - 1):
        B += int((np.max(np.abs(templates_m[i + 1:n]  - templates_m[i]),  axis=1) < r).sum())
        A += int((np.max(np.abs(templates_m1[i + 1:n] - templates_m1[i]), axis=1) < r).sum())
    if B == 0:
        return np.nan
    return float(-np.log(A / B)) if A > 0 else np.nan


def _detect_postprandial_excursions(
    g: np.ndarray,
    min_rise: float = 15.0,
    lookforward: int = 6,
    min_gap: int = 24,
) -> list:
    """Return onset indices where glucose rises ≥min_rise within `lookforward` readings."""
    events, i = [], 0
    while i < len(g) - lookforward:
        if g[i:i + lookforward].max() - g[i] >= min_rise:
            events.append(i)
            i += min_gap
        else:
            i += 1
    return events


def _postprandial_auc_and_peak_time(
    g: np.ndarray,
    onsets: list,
    window: int = 24,
    interval_min: float = 5.0,
) -> tuple:
    """Return (mean AUC above baseline over 120 min, mean time-to-peak in min)."""
    trapz = np.trapezoid
    aucs, peaks = [], []
    for i in onsets:
        if i + window >= len(g):
            continue
        baseline = g[i]
        seg      = g[i:i + window]
        aucs.append(float(trapz((seg - baseline).clip(min=0), dx=interval_min)))
        peaks.append(float(np.argmax(seg)) * interval_min)
    return (float(np.mean(aucs)) if aucs else np.nan,
            float(np.mean(peaks)) if peaks else np.nan)


def _overnight_slope(g: pd.Series, ts: pd.Series) -> tuple:
    """Mean and SD of per-night linear glucose slope (mg/dL/hr) between midnight and 06:00."""
    hour  = ts.dt.hour
    dates = ts.dt.date.unique()
    slopes = []
    for d in dates:
        mask = (hour >= 0) & (hour < 6) & (ts.dt.date == d)
        gn   = g[mask.values].values
        if len(gn) < 6:
            continue
        x = np.arange(len(gn)) * (5 / 60)
        slopes.append(float(np.polyfit(x, gn, 1)[0]))
    return ((float(np.mean(slopes)), float(np.std(slopes))) if slopes
            else (np.nan, np.nan))


def _dawn_phenomenon(g: pd.Series, ts: pd.Series) -> float:
    """Mean of (glucose at 08:00) − (nadir between 02:00–04:00) across days."""
    hour  = ts.dt.hour
    dates = ts.dt.date.unique()
    values = []
    for d in dates:
        nadir_mask   = (hour >= 2) & (hour < 4)  & (ts.dt.date == d)
        morning_mask = (hour == 8)                & (ts.dt.date == d)
        gn = g[nadir_mask.values]
        gm = g[morning_mask.values]
        if len(gn) == 0 or len(gm) == 0:
            continue
        values.append(float(gm.mean()) - float(gn.min()))
    return float(np.mean(values)) if values else np.nan


def _circadian_features(g: pd.Series, ts: pd.Series) -> tuple:
    """Amplitude and peak-time (hours, 0–24) of the dominant 24 h glucose component via FFT."""
    slot = ((ts.dt.hour * 60 + ts.dt.minute) // 5).clip(0, 287).values
    slot_means = np.full(288, np.nan)
    for s in range(288):
        vals = g.values[slot == s]
        if len(vals) > 0:
            slot_means[s] = vals.mean()
    if np.isnan(slot_means).any():
        x = np.arange(288)
        valid = ~np.isnan(slot_means)
        if valid.sum() < 10:
            return np.nan, np.nan
        slot_means = np.interp(x, x[valid], slot_means[valid])
    fft   = np.fft.rfft(slot_means - slot_means.mean())
    amp   = float(2 * abs(fft[1]) / 288)
    phase = float((-np.angle(fft[1]) / (2 * np.pi) * 24) % 24)
    return amp, phase


def _morning_rise_rate(g: pd.Series, ts: pd.Series) -> float:
    """Mean rate of change (mg/dL/min) between 06:00 and 09:00."""
    mask = ts.dt.hour.between(6, 8)
    roc  = g.diff()[mask.values].dropna()
    return float(roc.mean() / 5.0) if len(roc) > 0 else np.nan


def _lbgi(g: pd.Series) -> float:
    """Low Blood Glucose Index — quantifies risk-weighted hypoglycemic burden."""
    glucose = np.clip(g.values.astype(float), 1, None)
    f    = 1.794 * (np.log(glucose) ** 1.026 - 1.861)
    risk = np.where(f < 0, 10 * f ** 2, 0.0)
    return float(risk.mean())


def _interday_variability(g: pd.Series, ts: pd.Series) -> float:
    """Mean SD of glucose across days at each 5-minute time slot of day."""
    slot  = ((ts.dt.hour * 60 + ts.dt.minute) // 5).clip(0, 287).values
    dates = ts.dt.date.values
    sds   = []
    for s in range(288):
        mask   = slot == s
        if mask.sum() < 2:
            continue
        by_day = {}
        for val, d in zip(g.values[mask], dates[mask]):
            by_day.setdefault(d, []).append(val)
        day_means = [np.mean(v) for v in by_day.values()]
        if len(day_means) >= 2:
            sds.append(float(np.std(day_means, ddof=1)))
    return float(np.mean(sds)) if sds else np.nan


def _autocorrelation_lag(g: pd.Series, lag: int = 5) -> float:
    """Pearson autocorrelation at the given reading lag (default 5 = 25 min)."""
    arr = g.values
    if len(arr) <= lag:
        return np.nan
    return float(np.corrcoef(arr[:-lag], arr[lag:])[0, 1])


def _episode_duration_mean(g: pd.Series, threshold: float, above: bool = True,
                           interval_min: float = 5.0) -> float:
    """Mean duration (minutes) of contiguous episodes above/below threshold."""
    cond = (g > threshold) if above else (g < threshold)
    runs = (cond != cond.shift()).cumsum()
    durations = [
        len(grp) * interval_min
        for _, grp in cond.groupby(runs)
        if grp.iloc[0] and len(grp) >= _EVENT_MIN_READINGS
    ]
    return float(np.mean(durations)) if durations else 0.0


def _hyper_recovery_slope(g: pd.Series, threshold: float = 180.0,
                          interval_min: float = 5.0) -> float:
    """Mean slope (mg/dL/min) of the descending limb of hyperglycemic episodes."""
    cond  = g > threshold
    runs  = (cond != cond.shift()).cumsum()
    slopes = []
    for _, grp in g.groupby(runs):
        if not grp.iloc[0] > threshold or len(grp) < 4:
            continue
        pk = int(grp.values.argmax())
        desc = grp.values[pk:]
        if len(desc) < 3:
            continue
        x = np.arange(len(desc)) * interval_min
        slopes.append(float(np.polyfit(x, desc, 1)[0]))
    return float(np.mean(slopes)) if slopes else np.nan


def _glucose_drop_after_peak_60min(g: np.ndarray, readings_ahead: int = 12,
                                   min_peak: float = 140.0) -> float:
    """Mean (peak − glucose 60 min after peak) across significant peaks."""
    drops = []
    for i in range(1, len(g) - readings_ahead):
        if g[i] >= min_peak and g[i] > g[i - 1] and g[i] > g[i + 1]:
            drops.append(float(g[i] - g[i + readings_ahead]))
    return float(np.mean(drops)) if drops else np.nan


def _excursion_auc_normalized(g: pd.Series, threshold: float = 180.0,
                              interval_min: float = 5.0) -> float:
    """AUC above threshold divided by peak excess height — isolates excursion duration shape."""
    trapz = np.trapezoid
    cond  = g > threshold
    runs  = (cond != cond.shift()).cumsum()
    ratios = []
    for _, grp in g.groupby(runs):
        if not grp.iloc[0] > threshold:
            continue
        peak_excess = float(grp.max()) - threshold
        if peak_excess <= 0:
            continue
        auc = float(trapz((grp.values - threshold).clip(min=0), dx=interval_min))
        ratios.append(auc / peak_excess)
    return float(np.mean(ratios)) if ratios else np.nan


def _rise_fall_ratio(g: np.ndarray, onsets: list, max_window: int = 48) -> float:
    """
    For each excursion: (readings onset→peak) / (readings peak→return-to-baseline).
    Ratio < 1 = falls faster than rises (healthy); > 1 = prolonged clearance (IR).
    """
    ratios = []
    for i in onsets:
        end = min(i + max_window, len(g))
        seg = g[i:end]
        if len(seg) < 6:
            continue
        pk = int(np.argmax(seg))
        if pk == 0 or pk >= len(seg) - 1:
            continue
        target = g[i] + 10.0
        fall_time = None
        for j in range(pk + 1, len(seg)):
            if seg[j] <= target:
                fall_time = j - pk
                break
        if fall_time and fall_time > 0 and pk > 0:
            ratios.append(pk / fall_time)
    return float(np.mean(ratios)) if ratios else np.nan


def _roc_positive_negative_ratio(g: pd.Series) -> float:
    """Mean positive ROC / |mean negative ROC| — >1 means rises outpace falls."""
    roc = g.diff().dropna().values
    pos = roc[roc > 0]
    neg = roc[roc < 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(pos.mean() / abs(neg.mean()))


def _fasting_trend_slope(g: pd.Series, ts: pd.Series, min_days: int = 5) -> float:
    """Linear slope of daily mean fasting glucose (03:00–06:00) over the study period (mg/dL/day)."""
    hour  = ts.dt.hour
    dates = sorted(ts.dt.date.unique())
    daily = []
    for d in dates:
        mask = (hour >= 3) & (hour < 6) & (ts.dt.date == d)
        gn = g[mask.values]
        daily.append(float(gn.mean()) if len(gn) >= 3 else np.nan)
    valid_idx  = [i for i, v in enumerate(daily) if not np.isnan(v)]
    if len(valid_idx) < min_days:
        return np.nan
    x = np.array(valid_idx, dtype=float)
    y = np.array([daily[i] for i in valid_idx])
    return float(np.polyfit(x, y, 1)[0])


def _meal_count_per_day(g: pd.Series, ts: pd.Series,
                        min_rise: float = 30.0,
                        rise_window: int = 6,
                        min_duration: int = 12) -> float:
    """Estimated mean meal events per day: excursions with rise >30 mg/dL in 30 min, lasting >60 min."""
    arr = g.values
    i, events = 0, 0
    half_rise = min_rise / 2
    while i < len(arr) - max(rise_window, min_duration):
        if arr[i:i + rise_window].max() - arr[i] >= min_rise:
            threshold = arr[i] + half_rise
            duration  = sum(1 for j in range(i, min(i + 3 * min_duration, len(arr)))
                            if arr[j] > threshold)
            if duration >= min_duration:
                events += 1
                i += min_duration
                continue
        i += 1
    if len(ts) < 2:
        return np.nan
    total_days = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400
    return float(events / total_days) if total_days >= 1 else np.nan


def _glucose_range_night_vs_day(g: pd.Series, ts: pd.Series) -> float:
    """(day_max − day_min) / (night_max − night_min): IR → wide day range, narrow night range."""
    hour    = ts.dt.hour
    is_day  = hour.between(_DAY_START, _DAY_END - 1)
    g_day   = g[is_day.values]
    g_night = g[~is_day.values]
    if len(g_day) == 0 or len(g_night) == 0:
        return np.nan
    night_range = float(g_night.max() - g_night.min())
    if night_range == 0:
        return np.nan
    return float((g_day.max() - g_day.min()) / night_range)


def _find_peaks_prominent(arr: np.ndarray, min_prominence: float = 30.0):
    """Return (peak_indices, left_bases, right_bases) using scipy find_peaks."""
    from scipy.signal import find_peaks
    peaks, props = find_peaks(arr, prominence=min_prominence)
    return peaks, props.get("left_bases", np.array([])), props.get("right_bases", np.array([]))


def _spike_asymmetry_ratio(g: pd.Series, min_prominence: float = 30.0) -> float:
    """Mean (fall duration / rise duration) per prominent peak. >1 → slow clearance."""
    peaks, left_bases, right_bases = _find_peaks_prominent(g.values, min_prominence)
    if len(peaks) == 0:
        return np.nan
    ratios = []
    for pk, lb, rb in zip(peaks, left_bases, right_bases):
        rise = pk - lb
        fall = rb - pk
        if rise > 0 and fall > 0:
            ratios.append(fall / rise)
    return float(np.mean(ratios)) if ratios else np.nan


def _postprandial_plateau_duration(g: pd.Series, min_prominence: float = 30.0,
                                   plateau_pct: float = 0.90,
                                   interval_min: float = 5.0) -> float:
    """Mean minutes each peak spends within 10% of its peak value."""
    peaks, left_bases, right_bases = _find_peaks_prominent(g.values, min_prominence)
    if len(peaks) == 0:
        return np.nan
    arr = g.values
    durations = []
    for pk, lb, rb in zip(peaks, left_bases, right_bases):
        threshold = arr[pk] * plateau_pct
        window    = arr[lb:rb + 1]
        durations.append(float(np.sum(window > threshold)) * interval_min)
    return float(np.mean(durations)) if durations else np.nan


def _excursion_frequency(g: pd.Series, ts: pd.Series, pct: float = 75.0) -> float:
    """Upward crossings per day of the patient's own 75th-percentile glucose threshold."""
    arr       = g.values
    threshold = float(np.percentile(arr, pct))
    above     = arr > threshold
    crossings = int(np.sum((~above[:-1]) & above[1:]))
    total_days = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400
    return float(crossings / total_days) if total_days >= 1 else np.nan


def _nocturnal_nadir_instability(g: pd.Series, ts: pd.Series) -> float:
    """CV (std/mean × 100) of glucose during 00:00–06:00 — nocturnal floor variability."""
    mask   = ts.dt.hour.between(0, 5)
    g_night = g[mask.values]
    if len(g_night) < 10:
        return np.nan
    mean_val = float(g_night.mean())
    return float(g_night.std() / mean_val * 100) if mean_val > 0 else np.nan


def _interpeak_interval_cv(g: pd.Series, ts: pd.Series,
                            min_prominence: float = 30.0) -> float:
    """CV of time (minutes) between successive prominent glucose peaks."""
    peaks, _, _ = _find_peaks_prominent(g.values, min_prominence)
    if len(peaks) < 3:
        return np.nan
    peak_times = ts.iloc[peaks].values
    intervals  = (np.diff(peak_times) / np.timedelta64(1, "m")).astype(float)
    mean_iv    = float(intervals.mean())
    return float(intervals.std() / mean_iv) if mean_iv > 0 else np.nan


def _spectral_power_ratio(g: pd.Series, interval_min: float = 5.0) -> float:
    """
    Reactive-band (1–4 h) / meal-band (6–12 h) FFT power ratio.
    High ratio → fast oscillations dominate over meal-driven cycles.
    """
    arr = g.values - g.mean()
    N   = len(arr)
    if N < 24 * 12:   # need ≥ 2 days of data
        return np.nan
    power  = np.abs(np.fft.rfft(arr)) ** 2
    freqs  = np.fft.rfftfreq(N, d=interval_min / 60)   # cycles per hour
    with np.errstate(divide="ignore", invalid="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    meal_power     = power[(periods >= 6)  & (periods <= 12)].sum()
    reactive_power = power[(periods >= 1)  & (periods <= 4)].sum()
    return float(reactive_power / meal_power) if meal_power > 0 else np.nan


def _rolling_iqr_std(g: pd.Series, window_readings: int = 24,
                     step: int = 6) -> float:
    """
    Std of the rolling 2h IQR across the full trace.
    Low = stable glucose corridor; high = erratic within-window variability.
    """
    arr  = g.values
    iqrs = [
        float(np.percentile(arr[i:i + window_readings], 75) -
              np.percentile(arr[i:i + window_readings], 25))
        for i in range(0, len(arr) - window_readings + 1, step)
    ]
    return float(np.std(iqrs)) if len(iqrs) >= 2 else np.nan


def _predawn_drift(g: pd.Series, ts: pd.Series) -> float:
    """
    Mean(00:00–03:30) − Mean(22:00–23:59 of the previous evening), averaged across nights.
    Captures hepatic glucose leak before the classical dawn phenomenon window.
    """
    from datetime import timedelta
    hour, minute = ts.dt.hour, ts.dt.minute
    dates = sorted(ts.dt.date.unique())
    drifts = []
    for d in dates:
        eve_mask     = (hour >= 22) & (ts.dt.date == d)
        predawn_mask = ((hour < 3) | ((hour == 3) & (minute <= 30))) & (ts.dt.date == d)
        # Evening is the prior calendar day's 22:00–23:59
        prev_d       = d - timedelta(days=1)
        prev_mask    = (hour >= 22) & (ts.dt.date == prev_d)
        g_eve    = g[prev_mask.values]
        g_predawn= g[predawn_mask.values]
        if len(g_eve) < 3 or len(g_predawn) < 3:
            continue
        drifts.append(float(g_predawn.mean()) - float(g_eve.mean()))
    return float(np.mean(drifts)) if drifts else np.nan


def _intermeal_nadir_mean(g: pd.Series, ts: pd.Series,
                          min_drop: float = 20.0) -> float:
    """
    Mean glucose at daytime (10:00–18:00) local minima that lie >= min_drop below
    surrounding peaks. Captures the inter-meal glucose floor.
    """
    hour     = ts.dt.hour
    mask     = hour.between(10, 17)
    g_day    = g[mask.values].values
    if len(g_day) < 6:
        return np.nan
    nadirs = []
    for i in range(1, len(g_day) - 1):
        if g_day[i] < g_day[i - 1] and g_day[i] < g_day[i + 1]:
            # Check if valley is >= min_drop below the higher of its neighbours
            local_peak = max(g_day[max(0, i - 12):i].max() if i >= 12 else g_day[:i].max(),
                             g_day[i + 1:min(len(g_day), i + 12)].max() if i + 12 <= len(g_day) else g_day[i + 1:].max())
            if local_peak - g_day[i] >= min_drop:
                nadirs.append(g_day[i])
    return float(np.mean(nadirs)) if nadirs else np.nan


def _morning_rise_onset_hour(g: pd.Series, ts: pd.Series,
                              threshold_rise: float = 10.0,
                              ceiling_hour: float = 9.0) -> float:
    """
    Mean fractional hour at which glucose first crosses (overnight_nadir + threshold_rise)
    in the 02:00–09:00 window each day. Earlier onset → more IR.
    """
    hour, minute = ts.dt.hour, ts.dt.minute
    dates = sorted(ts.dt.date.unique())
    onsets = []
    for d in dates:
        window_mask = hour.between(2, 8) & (ts.dt.date == d)
        g_win = g[window_mask.values].values
        t_win = ts[window_mask.values]
        if len(g_win) < 6:
            continue
        nadir     = g_win.min()
        threshold = nadir + threshold_rise
        onset     = ceiling_hour
        for i, val in enumerate(g_win):
            if val >= threshold:
                onset = t_win.iloc[i].hour + t_win.iloc[i].minute / 60
                break
        onsets.append(onset)
    return float(np.mean(onsets)) if onsets else np.nan


def _daytime_glucose_floor_pct(g: pd.Series, ts: pd.Series,
                                offset: float = 5.0) -> float:
    """
    % of 10:00–18:00 readings below (patient_overnight_mean + offset).
    Measures whether daytime glucose ever returns to the patient's own overnight baseline.
    """
    hour = ts.dt.hour
    overnight_mask = (hour >= 22) | (hour < 6)
    daytime_mask   = hour.between(10, 17)
    g_overnight    = g[overnight_mask.values]
    g_daytime      = g[daytime_mask.values]
    if len(g_overnight) < 10 or len(g_daytime) < 10:
        return np.nan
    threshold = float(g_overnight.mean()) + offset
    return float((g_daytime < threshold).mean() * 100)


def _dawn_to_noon_auc_excess(g: pd.Series, ts: pd.Series,
                              interval_min: float = 5.0) -> float:
    """
    AUC(05:00–12:00) minus the flat-baseline AUC at the patient's 04:00 glucose level.
    Captures the morning glucose excess above the pre-dawn starting point.
    """
    trapz = np.trapezoid
    hour  = ts.dt.hour
    dates = sorted(ts.dt.date.unique())
    excesses = []
    for d in dates:
        base_mask   = (hour == 4) & (ts.dt.date == d)
        window_mask = hour.between(5, 11) & (ts.dt.date == d)
        g_base   = g[base_mask.values]
        g_window = g[window_mask.values].values
        if len(g_base) == 0 or len(g_window) < 6:
            continue
        baseline = float(g_base.mean())
        auc      = float(trapz(g_window, dx=interval_min))
        expected = baseline * len(g_window) * interval_min
        excesses.append(auc - expected)
    return float(np.mean(excesses)) if excesses else np.nan


def _evening_convergence_rate(g: pd.Series, ts: pd.Series) -> float:
    """
    Mean linear slope (mg/dL/hr) of glucose during 18:30–20:30 per day.
    Captures the post-dinner descent rate where the two groups converge.
    """
    hour, minute = ts.dt.hour, ts.dt.minute
    dates = sorted(ts.dt.date.unique())
    slopes = []
    for d in dates:
        mask = ((hour == 18) & (minute >= 30) |
                (hour == 19) |
                (hour == 20) & (minute <= 30)) & (ts.dt.date == d)
        g_win = g[mask.values].values
        if len(g_win) < 4:
            continue
        x = np.arange(len(g_win)) * (5 / 60)
        slope = float(np.polyfit(x, g_win, 1)[0])
        slopes.append(max(slope, 0.0) if slope >= 0 else slope)
    return float(np.mean(slopes)) if slopes else np.nan


def _glucose_plateau_index(g: pd.Series, ts: pd.Series,
                            window_hours: int = 2) -> float:
    """
    For each day, find the quietest 2h daytime window (min SD across non-overlapping 2h
    windows in 08:00–18:00). Average across days. Low value = patient can sustain
    stable inter-meal periods.
    """
    hour  = ts.dt.hour
    dates = sorted(ts.dt.date.unique())
    window_readings = window_hours * 12  # 12 readings/hour at 5-min intervals
    min_sds = []
    for d in dates:
        mask  = hour.between(8, 17) & (ts.dt.date == d)
        g_day = g[mask.values].values
        if len(g_day) < window_readings:
            continue
        sds = [
            float(g_day[i:i + window_readings].std())
            for i in range(0, len(g_day) - window_readings + 1, window_readings)
        ]
        if sds:
            min_sds.append(min(sds))
    return float(np.mean(min_sds)) if min_sds else np.nan


def extract_cgm_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute a rich feature vector from a patient's full CGM trace.

    Parameters
    ----------
    df : DataFrame with columns [timestamp (datetime64, UTC), glucose (float)]

    Returns
    -------
    dict of feature_name -> float
    """
    _require_columns(df)

    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    g  = df["glucose"]
    ts = pd.to_datetime(df["timestamp"], utc=True)

    # ── Basic statistics ──────────────────────────────────────────────────────
    feat: Dict[str, float] = {
        "cgm_mean":   round(float(g.mean()), 2),
        "cgm_std":    round(float(g.std()),  2),
        "cgm_median": round(float(g.median()), 2),
        "cgm_min":    round(float(g.min()),  2),
        "cgm_max":    round(float(g.max()),  2),
        "cgm_cv_pct": round(float(g.std() / g.mean() * 100) if g.mean() != 0 else 0.0, 2),
    }

    # ── Time-in-range variants ────────────────────────────────────────────────
    feat["cgm_tir_70_180"]  = round(_tir(g, _HYPO_L1, _TARGET_H), 2)
    feat["cgm_tir_70_140"]  = round(_tir(g, _HYPO_L1, 140), 2)
    feat["cgm_tbr_lt70"]    = round(_tir(g, 0,        _HYPO_L1 - 0.001), 2)
    feat["cgm_tbr_lt54"]    = round(_tir(g, 0,        _HYPO_L2 - 0.001), 2)
    feat["cgm_tar_gt180"]   = round(_tir(g, _TARGET_H + 0.001, 9999), 2)
    feat["cgm_tar_gt250"]   = round(_tir(g, _HYPER_L2 + 0.001, 9999), 2)

    # ── Glycemic variability indices ──────────────────────────────────────────
    feat["cgm_j_index"]     = round(0.001 * (g.mean() + g.std()) ** 2, 3)
    feat["cgm_mage"]        = round(_mage(g), 2)
    feat["cgm_iqr"]         = round(float(g.quantile(0.75) - g.quantile(0.25)), 2)

    # Rate-of-change stats
    roc = _rate_of_change(g).dropna()
    feat["cgm_roc_mean_abs"] = round(float(roc.abs().mean()), 3)
    feat["cgm_roc_std"]      = round(float(roc.std()), 3)
    feat["cgm_roc_pct95"]    = round(float(roc.abs().quantile(0.95)), 3)

    # ── Event counts ──────────────────────────────────────────────────────────
    feat["cgm_n_hypo_l1_events"]  = _count_events(g < _HYPO_L1)
    feat["cgm_n_hypo_l2_events"]  = _count_events(g < _HYPO_L2)
    feat["cgm_n_hyper_l1_events"] = _count_events(g > _TARGET_H)
    feat["cgm_n_hyper_l2_events"] = _count_events(g > _HYPER_L2)

    # ── Area under/over curve ─────────────────────────────────────────────────
    feat["cgm_auc_above_180"] = round(_auc_above(g, _TARGET_H), 1)
    feat["cgm_auc_above_140"] = round(_auc_above(g, 140.0),     1)
    feat["cgm_auc_above_70"]  = round(_auc_above(g, _HYPO_L1),  1)

    # ── Circadian split ───────────────────────────────────────────────────────
    hour = ts.dt.hour
    is_day   = hour.between(_DAY_START, _DAY_END - 1)
    g_day    = g[is_day.values]
    g_night  = g[~is_day.values]

    feat["cgm_day_mean"]          = round(float(g_day.mean())   if len(g_day)   > 0 else np.nan, 2)
    feat["cgm_night_mean"]        = round(float(g_night.mean()) if len(g_night) > 0 else np.nan, 2)
    feat["cgm_day_night_diff"]    = round(feat["cgm_day_mean"] - feat["cgm_night_mean"], 2)
    feat["cgm_night_tbr_lt70_pct"]= round(_tir(g_night, 0, _HYPO_L1 - 0.001) if len(g_night) > 0 else np.nan, 2)

    # ── Fasting / early-morning window (04:00–08:00) ──────────────────────────
    is_fasting = hour.between(4, 7)
    g_fasting  = g[is_fasting.values]
    feat["cgm_fasting_mean"]   = round(float(g_fasting.mean())   if len(g_fasting) > 1 else np.nan, 2)
    feat["cgm_fasting_std"]    = round(float(g_fasting.std())    if len(g_fasting) > 1 else np.nan, 2)
    feat["cgm_fasting_median"] = round(float(g_fasting.median()) if len(g_fasting) > 1 else np.nan, 2)

    # ── Post-wake rise (08:00–12:00 — proxy for morning glucose surge) ────────
    is_morning = hour.between(8, 11)
    g_morning  = g[is_morning.values]
    feat["cgm_morning_mean"] = round(float(g_morning.mean()) if len(g_morning) > 0 else np.nan, 2)
    feat["cgm_morning_peak"] = round(float(g_morning.max())  if len(g_morning) > 0 else np.nan, 2)

    # ── Sensor coverage ───────────────────────────────────────────────────────
    if len(ts) > 1:
        total_minutes    = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 60
        expected_readings = total_minutes / 5
        feat["cgm_wear_fraction"] = round(len(g) / expected_readings if expected_readings > 0 else np.nan, 3)
    else:
        feat["cgm_wear_fraction"] = np.nan

    feat["cgm_n_readings"] = len(g)

    # ── Advanced features (LLM-suggested) ────────────────────────────────────
    feat["cgm_glucose_entropy"] = round(_sample_entropy(g.values), 4)

    onsets = _detect_postprandial_excursions(g.values)
    pp_auc, pp_peak = _postprandial_auc_and_peak_time(g.values, onsets)
    feat["cgm_postprandial_auc_2h"]        = round(pp_auc,  1) if not np.isnan(pp_auc)  else np.nan
    feat["cgm_postprandial_peak_time_mean"]= round(pp_peak, 1) if not np.isnan(pp_peak) else np.nan

    slope_mean, slope_std = _overnight_slope(g, ts)
    feat["cgm_overnight_slope_mean"] = round(slope_mean, 3) if not np.isnan(slope_mean) else np.nan
    feat["cgm_overnight_slope_std"]  = round(slope_std,  3) if not np.isnan(slope_std)  else np.nan

    feat["cgm_dawn_phenomenon_magnitude"] = round(_dawn_phenomenon(g, ts), 2)

    circ_amp, circ_phase = _circadian_features(g, ts)
    feat["cgm_circadian_amplitude"]   = round(circ_amp,   2) if not np.isnan(circ_amp)   else np.nan
    feat["cgm_circadian_phase_hours"] = round(circ_phase, 2) if not np.isnan(circ_phase) else np.nan

    feat["cgm_glucose_rise_rate_morning"] = round(_morning_rise_rate(g, ts), 4)
    feat["cgm_lbgi"]                      = round(_lbgi(g), 3)
    feat["cgm_interday_variability"]      = round(_interday_variability(g, ts), 2)
    feat["cgm_autocorr_lag5"]             = round(_autocorrelation_lag(g), 4)
    feat["cgm_hyper_episode_duration_mean"] = round(_episode_duration_mean(g, _TARGET_H), 1)

    # ── New hypotheses: clearance dynamics, asymmetry, fasting shape, structure ──
    feat["cgm_hyper_recovery_slope"]       = round(_hyper_recovery_slope(g), 4)
    feat["cgm_glucose_drop_after_peak_60"] = round(_glucose_drop_after_peak_60min(g.values), 2)
    feat["cgm_excursion_auc_normalized"]   = round(_excursion_auc_normalized(g), 2)
    feat["cgm_rise_fall_ratio"]            = round(_rise_fall_ratio(g.values, onsets), 3)
    feat["cgm_roc_positive_negative_ratio"]= round(_roc_positive_negative_ratio(g), 4)
    feat["cgm_fasting_glucose_trend_slope"]= round(_fasting_trend_slope(g, ts), 4)
    feat["cgm_fasting_cv"] = (
        round(feat["cgm_fasting_std"] / feat["cgm_fasting_mean"] * 100, 2)
        if not np.isnan(feat.get("cgm_fasting_mean", np.nan)) and feat.get("cgm_fasting_mean", 0) > 0
        else np.nan
    )
    feat["cgm_meal_count_estimated"]       = round(_meal_count_per_day(g, ts), 3)
    feat["cgm_glucose_range_night_vs_day"] = round(_glucose_range_night_vs_day(g, ts), 3)

    # ── Profile-driven features (from stratified 24h CGM analysis) ────────────
    feat["cgm_predawn_drift"]             = round(_predawn_drift(g, ts), 2)
    feat["cgm_intermeal_nadir_mean"]      = round(_intermeal_nadir_mean(g, ts), 2)
    feat["cgm_morning_rise_onset_hour"]   = round(_morning_rise_onset_hour(g, ts), 3)
    feat["cgm_daytime_glucose_floor_pct"] = round(_daytime_glucose_floor_pct(g, ts), 2)
    feat["cgm_dawn_to_noon_auc_excess"]   = round(_dawn_to_noon_auc_excess(g, ts), 1)
    feat["cgm_evening_convergence_rate"]  = round(_evening_convergence_rate(g, ts), 4)
    feat["cgm_glucose_plateau_index"]     = round(_glucose_plateau_index(g, ts), 2)

    # ── Image-analysis-suggested features (visual LLM, round 3) ─────────────
    feat["cgm_spike_asymmetry_ratio"]        = round(_spike_asymmetry_ratio(g), 3)
    feat["cgm_postprandial_plateau_duration"]= round(_postprandial_plateau_duration(g), 1)
    feat["cgm_excursion_frequency"]          = round(_excursion_frequency(g, ts), 3)
    feat["cgm_nocturnal_nadir_instability"]  = round(_nocturnal_nadir_instability(g, ts), 2)
    feat["cgm_interpeak_interval_cv"]        = round(_interpeak_interval_cv(g, ts), 3)
    feat["cgm_spectral_power_ratio"]         = round(_spectral_power_ratio(g), 4)
    feat["cgm_rolling_iqr_std"]              = round(_rolling_iqr_std(g), 2)

    return feat
