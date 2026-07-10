"""
gradepace.py — shared engine for the GradePace tool.
Used by BOTH the Colab notebook (testing ground) and the web app (deployment).
Pure pandas/numpy/stdlib only. All tunables have defaults; the notebook's
config cell overrides just the dials that matter.
"""
import os
import datetime
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

# =====================================================================
# BAND DEFINITIONS + INTERNAL DEFAULTS (override via function args)
# =====================================================================
GRADE_BANDS = [
    (float("-inf"), -15.0, "1. Deep Descent (< -15%)"),
    (-15.0, -8.0, "2. Steep Descent (-15% to -8%)"),
    (-8.0, -2.0, "3. Rolling Descent (-8% to -2%)"),
    (-2.0, 2.0, "4. Flat / Undulating (-2% to 2%)"),
    (2.0, 6.0, "5. Gentle Climb (2% to 6%)"),
    (6.0, 12.0, "6. Steady Climb (6% to 12%)"),
    (12.0, 25.0, "7. Steep Climb (12% to 25%)"),
    (25.0, float("inf"), "8. Extreme Wall (> 25%)"),
]

# Altitude drag scales with how aerobically limited each band is:
# climbs feel the full effect, descents (mechanically limited) very little.
ALTITUDE_BAND_WEIGHTS = {
    "1. Deep Descent (< -15%)": 0.10,
    "2. Steep Descent (-15% to -8%)": 0.20,
    "3. Rolling Descent (-8% to -2%)": 0.40,
    "4. Flat / Undulating (-2% to 2%)": 0.80,
    "5. Gentle Climb (2% to 6%)": 1.00,
    "6. Steady Climb (6% to 12%)": 1.00,
    "7. Steep Climb (12% to 25%)": 1.00,
    "8. Extreme Wall (> 25%)": 1.00,
}

GEARS = ["fast", "median", "slow"]

DEFAULTS = dict(
    smooth_window=30,        # rolling samples for history smoothing
    course_smooth_window=15, # rolling samples for course grade smoothing
    grade_min=-35.0, grade_max=45.0,
    pace_min=5.0, pace_max=35.0,   # min/mile sanity filter
    fatigue_onset_mile=1.0,
    stopped_speed_mps=0.5,   # below this point-to-point speed = stopped
)

M_TO_FT = 3.28084
M_TO_MI = 0.000621371


def assign_band(g):
    for low, high, label in GRADE_BANDS:
        if low <= g < high:
            return label
    return "4. Flat / Undulating (-2% to 2%)"


# =====================================================================
# FORMATTERS
# =====================================================================
def f_pace(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) or v <= 0:
        return "N/A"
    mins = int(v)
    secs = int(round((v - mins) * 60))
    if secs == 60:
        mins, secs = mins + 1, 0
    return f"{mins}:{secs:02d}"


def f_time(s):
    if s is None or (isinstance(s, float) and np.isnan(s)) or s < 0:
        return "N/A"
    hr, mn, sc = int(s // 3600), int((s % 3600) // 60), int(s % 60)
    return f"{hr}:{mn:02d}:{sc:02d}" if hr > 0 else f"{mn}:{sc:02d}"


def f_signed(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "N/A"
    sign = "-" if s < 0 else "+"
    a = abs(int(round(s)))
    if a >= 3600:
        return f"{sign}{a // 3600}:{(a % 3600) // 60:02d}:{a % 60:02d}"
    return f"{sign}{a // 60}:{a % 60:02d}"


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    return R * (2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


# =====================================================================
# STREAM CACHE (incremental harvest support)
# =====================================================================
STREAM_COLS = ["activity_id", "time_sec", "distance_meters", "altitude_meters", "heart_rate_bpm"]
META_COLS = ["activity_id", "name", "type", "start_date"]


def load_cache(cache_dir):
    sp = os.path.join(cache_dir, "gradepace_streams.parquet")
    mp = os.path.join(cache_dir, "gradepace_activities.parquet")
    streams = pd.read_parquet(sp) if os.path.exists(sp) else pd.DataFrame(columns=STREAM_COLS)
    meta = pd.read_parquet(mp) if os.path.exists(mp) else pd.DataFrame(columns=META_COLS)
    return streams, meta


def save_cache(streams, meta, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    streams.to_parquet(os.path.join(cache_dir, "gradepace_streams.parquet"), index=False)
    meta.to_parquet(os.path.join(cache_dir, "gradepace_activities.parquet"), index=False)


def missing_ids(activity_list, meta):
    cached = set(meta["activity_id"].astype("int64")) if len(meta) else set()
    return [a for a in activity_list if int(a["id"]) not in cached]


def merge_new(streams, meta, new_stream_rows, new_meta_rows):
    if new_stream_rows:
        streams = pd.concat([streams, pd.DataFrame(new_stream_rows)], ignore_index=True)
    if new_meta_rows:
        meta = pd.concat([meta, pd.DataFrame(new_meta_rows)], ignore_index=True)
        meta = meta.drop_duplicates(subset="activity_id", keep="last")
    return streams, meta


def _norm_type(t):
    """Normalize activity types so 'Trail Run', 'TrailRun', 'trailrun' all match."""
    return str(t).replace(" ", "").lower()


def activity_matches(act, activity_types):
    """True if a Strava activity dict matches any selected type.
    Prefers sport_type (which distinguishes TrailRun) over legacy type."""
    sel = {_norm_type(t) for t in activity_types}
    val = act.get("sport_type") or act.get("type") or ""
    return _norm_type(val) in sel


def filter_meta(meta, start_date, end_date, activity_types):
    m = meta.copy()
    m["start_date"] = pd.to_datetime(m["start_date"], utc=True, format="ISO8601")
    sel = {_norm_type(t) for t in activity_types}
    mask = (
        (m["start_date"].dt.date >= start_date)
        & (m["start_date"].dt.date <= end_date)
        & (m["type"].map(_norm_type).isin(sel))
    )
    return m.loc[mask]


def select_range(streams, meta, start_date, end_date, activity_types):
    m = filter_meta(meta, start_date, end_date, activity_types)
    ids = set(m["activity_id"].astype("int64"))
    return streams[streams["activity_id"].astype("int64").isin(ids)].copy(), len(ids)


# =====================================================================
# PROFILE ENGINE (history -> grade-band pace/HR quartiles + baseline)
# =====================================================================
def build_profile(raw, **over):
    p = {**DEFAULTS, **over}
    raw = raw.sort_values(["activity_id", "time_sec"]).reset_index(drop=True)
    g = raw.groupby("activity_id")
    roll = lambda s: s.rolling(window=p["smooth_window"], center=True, min_periods=5).mean()
    raw["smooth_dist"] = g["distance_meters"].transform(roll)
    raw["smooth_alt"] = g["altitude_meters"].transform(roll)
    raw["smooth_time"] = g["time_sec"].transform(roll)
    raw["smooth_hr"] = g["heart_rate_bpm"].transform(roll)

    df = raw.copy()
    gg = df.groupby("activity_id")
    df["delta_dist"] = gg["smooth_dist"].diff()
    df["delta_time"] = gg["smooth_time"].diff()
    df["delta_alt"] = gg["smooth_alt"].diff()
    df = df.dropna(subset=["delta_dist", "delta_time", "delta_alt"])
    df = df[(df["delta_dist"] > 0.2) & (df["delta_time"] > 0.2)].copy()

    df["speed_mps"] = df["delta_dist"] / df["delta_time"]
    df["grade"] = (df["delta_alt"] / df["delta_dist"]) * 100
    df["pace_min_per_mile"] = 26.8224 / df["speed_mps"]
    df = df[
        (df["grade"] >= p["grade_min"]) & (df["grade"] <= p["grade_max"])
        & (df["pace_min_per_mile"] >= p["pace_min"]) & (df["pace_min_per_mile"] <= p["pace_max"])
    ].copy()
    df["grade_band"] = df["grade"].apply(assign_band)

    rows = []
    for band, grp in df.groupby("grade_band"):
        pq = grp["pace_min_per_mile"].quantile([0.25, 0.5, 0.75])
        hq = grp["smooth_hr"].quantile([0.25, 0.5, 0.75])
        rows.append(dict(
            grade_band=band,
            pace_fast=pq[0.25], pace_median=pq[0.5], pace_slow=pq[0.75],
            hr_fast=hq[0.25], hr_median=hq[0.5], hr_slow=hq[0.75],
            n_samples=len(grp),
        ))
    profile = pd.DataFrame(rows).sort_values("grade_band").reset_index(drop=True)
    baseline_ft = float(raw["altitude_meters"].median()) * M_TO_FT
    return profile, baseline_ft


def get_gear_pace(profile, band, gear):
    """Band pace with nearest-band fallback for bands never trained in."""
    lut = profile.set_index("grade_band")
    col = f"pace_{gear}"
    if band in lut.index:
        return lut.loc[band, col]
    target_n = int(band.split(".")[0])
    nearest = min(lut.index, key=lambda b: abs(int(b.split(".")[0]) - target_n))
    return lut.loc[nearest, col]


# =====================================================================
# GPX PARSING
# =====================================================================
def parse_gpx(path_or_file):
    tree = ET.parse(path_or_file)
    root = tree.getroot()
    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    pts = root.findall(".//gpx:trkpt", ns) or root.findall(".//trkpt")
    def _find_hr(tp):
        """HR lives in watch-specific extension blocks; match any descendant tag
        named 'hr' regardless of namespace (Garmin/Coros/Suunto exports)."""
        for el in tp.iter():
            tag = el.tag.split("}")[-1].lower()
            if tag == "hr" and el.text:
                try:
                    return float(el.text)
                except ValueError:
                    return np.nan
        return np.nan

    recs = []
    for tp in pts:
        ele = tp.find("gpx:ele", ns)
        if ele is None:
            ele = tp.find("ele")
        tn = tp.find("gpx:time", ns)
        if tn is None:
            tn = tp.find("time")
        ts = None
        if tn is not None and tn.text:
            ts = datetime.datetime.fromisoformat(tn.text.strip().replace("Z", "+00:00"))
        recs.append(dict(
            lat=float(tp.get("lat")), lon=float(tp.get("lon")),
            ele=float(ele.text) if ele is not None else 0.0,
            timestamp=ts,
            hr=_find_hr(tp),
        ))
    df = pd.DataFrame(recs)
    df.attrs["has_telemetry"] = df["timestamp"].notna().mean() > 0.9 if len(df) else False
    return df


# =====================================================================
# COURSE SIMULATOR (+ optional telemetry attachment)
# =====================================================================
def simulate(df_gpx, profile, baseline_ft,
             fatigue_rate=1.0, altitude_rate=2.0,
             allow_altitude_benefit=False, **over):
    p = {**DEFAULTS, **over}
    df = df_gpx.copy()
    if "hr" not in df.columns:
        df["hr"] = np.nan
    df["delta_dist"] = haversine(df["lat"].shift(), df["lon"].shift(), df["lat"], df["lon"])
    df.loc[df.index[0], "delta_dist"] = 0.0
    df["delta_ele"] = df["ele"].diff().fillna(0.0)
    df["cum_dist_miles"] = df["delta_dist"].cumsum() * M_TO_MI
    df["delta_dist_miles"] = df["delta_dist"] * M_TO_MI
    df["calc_grade"] = np.where(df["delta_dist"] > 0.1, (df["delta_ele"] / df["delta_dist"]) * 100, 0.0)
    df["smoothed_grade"] = df["calc_grade"].rolling(
        p["course_smooth_window"], min_periods=1, center=True).mean().fillna(0.0)
    df["mile_bucket"] = np.floor(df["cum_dist_miles"]).astype(int) + 1
    df["grade_band"] = df["smoothed_grade"].apply(assign_band)

    # altitude drag (aerobic-weighted, relative to personal baseline)
    df["ele_ft"] = df["ele"] * M_TO_FT
    delta_kft = (df["ele_ft"] - baseline_ft) / 1000.0
    if not allow_altitude_benefit:
        delta_kft = np.maximum(0.0, delta_kft)
    df["aerobic_weight"] = df["grade_band"].map(ALTITUDE_BAND_WEIGHTS).fillna(1.0)
    df["alt_multiplier"] = 1.0 + delta_kft * df["aerobic_weight"] * (altitude_rate / 100.0)
    df["fatigue_multiplier"] = 1.0 + np.maximum(
        0, df["cum_dist_miles"] - p["fatigue_onset_mile"]) * (fatigue_rate / 100.0)

    pace_lut = {(b, g): get_gear_pace(profile, b, g)
                for b in df["grade_band"].unique() for g in GEARS}
    for gear in GEARS:
        df[f"base_pace_{gear}"] = df["grade_band"].map(lambda b: pace_lut[(b, gear)])
        df[f"sim_pace_{gear}"] = df[f"base_pace_{gear}"] * df["fatigue_multiplier"] * df["alt_multiplier"]
        df[f"pred_sec_{gear}"] = df[f"sim_pace_{gear}"] * 60.0 * df["delta_dist_miles"]
        fresh = df[f"base_pace_{gear}"] * 60.0 * df["delta_dist_miles"]
        df[f"raw_sec_{gear}"] = fresh  # band paces only: no fatigue/altitude adjustment
        df[f"fatigue_cost_{gear}"] = (df[f"base_pace_{gear}"]
                                      * (df["fatigue_multiplier"] - 1.0)) * 60.0 * df["delta_dist_miles"]
        df[f"altitude_cost_{gear}"] = df[f"pred_sec_{gear}"] - fresh - df[f"fatigue_cost_{gear}"]
        df[f"cum_pred_{gear}"] = df[f"pred_sec_{gear}"].cumsum()

    # telemetry (watch file) -> actual elapsed/moving/stopped
    df["has_telemetry"] = bool(df_gpx.attrs.get("has_telemetry", False))
    if df["has_telemetry"].iloc[0]:
        ts = pd.to_datetime(df_gpx["timestamp"])
        df["actual_elapsed"] = (ts - ts.iloc[0]).dt.total_seconds().values
        df["actual_delta"] = pd.Series(df["actual_elapsed"]).diff().fillna(0.0).values
        speed = df["delta_dist"] / pd.Series(df["actual_delta"]).replace(0, np.nan)
        df["stopped_sec"] = np.where(
            (speed < p["stopped_speed_mps"]) & (df["actual_delta"] > 0), df["actual_delta"], 0.0)
    else:
        df["actual_elapsed"] = np.nan
        df["actual_delta"] = np.nan
        df["stopped_sec"] = np.nan
    return df


# =====================================================================
# HISTORICAL STOP ANALYSIS (per-activity stopped time from cached streams)
# =====================================================================
def historical_stops(streams, stopped_speed_mps=None):
    """Per-activity distance + stopped time from cached streams.
    Counts both recorded-idle samples and pause gaps (big dt, no distance)."""
    thresh = stopped_speed_mps or DEFAULTS["stopped_speed_mps"]
    rows = []
    for aid, g in streams.groupby("activity_id"):
        g = g.sort_values("time_sec")
        dt = g["time_sec"].diff()
        dd = g["distance_meters"].diff()
        valid = dt > 0
        speed = dd[valid] / dt[valid]
        stopped = float(dt[valid][speed < thresh].sum())
        rows.append(dict(activity_id=aid,
                         miles=float(g["distance_meters"].max()) * M_TO_MI,
                         stopped_sec=stopped))
    return pd.DataFrame(rows)


# =====================================================================
# CALIBRATION INSIGHTS (post-run residuals -> dial suggestions)
# =====================================================================
def calibration_insights(df, fatigue_rate, altitude_rate):
    """Directional dial suggestions from mile-level residuals. Telemetry required."""
    if not df["has_telemetry"].iloc[0]:
        return []
    miles = []
    for m, g in df.groupby("mile_bucket"):
        mi = g["delta_dist_miles"].sum()
        if mi < 0.5:
            continue
        moving = g["actual_delta"].sum() - g["stopped_sec"].sum()
        pred = g["pred_sec_median"].sum()
        miles.append(dict(mile=m, resid_pct=(moving - pred) / pred * 100,
                          alt_cost=g["altitude_cost_median"].sum(),
                          cum=g["cum_dist_miles"].iloc[-1]))
    md = pd.DataFrame(miles)
    out = []
    if len(md) < 4:
        return ["Not enough full miles for calibration analysis."]
    hi = md[md["alt_cost"] > 15]
    lo = md[md["alt_cost"] <= 15]
    if len(hi) >= 2 and len(lo) >= 2:
        gap = hi["resid_pct"].median() - lo["resid_pct"].median()
        if abs(gap) > 2:
            direction = "raising" if gap > 0 else "lowering"
            out.append(
                f"High-altitude miles ran {gap:+.1f}% vs low-altitude miles (relative to targets). "
                f"Consider {direction} ALTITUDE_RATE_PER_1000FT from {altitude_rate}."
            )
        else:
            out.append(f"Altitude model held up well (high vs low altitude residual gap {gap:+.1f}%).")
    third = len(md) // 3
    early, late = md.iloc[:third], md.iloc[-third:]
    if third >= 2:
        drift = late["resid_pct"].median() - early["resid_pct"].median()
        if abs(drift) > 2:
            direction = "raising" if drift > 0 else "lowering"
            out.append(
                f"Late-race miles drifted {drift:+.1f}% vs early miles. "
                f"Consider {direction} FATIGUE_RATE from {fatigue_rate}."
            )
        else:
            out.append(f"Fatigue model held up well (late vs early residual drift {drift:+.1f}%).")
    return out


# =====================================================================
# RUN INSIGHTS (deterministic post-run analysis beyond dial calibration)
# =====================================================================
def run_insights(df):
    """Narrative-style observations about execution. Telemetry required."""
    if not df["has_telemetry"].iloc[0]:
        return []
    miles = []
    for m, g in df.groupby("mile_bucket"):
        mi = g["delta_dist_miles"].sum()
        if mi < 0.5:
            continue
        moving = g["actual_delta"].sum() - g["stopped_sec"].sum()
        pred = g["pred_sec_median"].sum()
        miles.append(dict(
            mile=int(m), var_pct=(moving - pred) / pred * 100,
            stop=g["stopped_sec"].sum(),
            is_desc=g["grade_band"].str.contains("Descent").mean() > 0.5,
            is_climb=g["grade_band"].str.contains("Climb|Wall").mean() > 0.5,
        ))
    md = pd.DataFrame(miles)
    out = []
    if len(md) < 2:
        return out

    best = md.loc[md["var_pct"].idxmin()]
    worst = md.loc[md["var_pct"].idxmax()]
    out.append(f"Best-executed mile: {int(best['mile'])} ({best['var_pct']:+.0f}% vs target). "
               f"Toughest: mile {int(worst['mile'])} ({worst['var_pct']:+.0f}%).")

    half = len(md) // 2
    fh = md.iloc[:half]["var_pct"].median()
    sh = md.iloc[half:]["var_pct"].median()
    style = "strengthened relative to plan late" if sh < fh else "faded relative to plan late"
    out.append(f"Execution split: first half {fh:+.1f}% vs target, second half {sh:+.1f}% — {style}.")

    climbs, descs = md[md["is_climb"]], md[md["is_desc"]]
    if len(climbs) >= 2 and len(descs) >= 2:
        out.append(f"Terrain execution: climbing miles ran {climbs['var_pct'].median():+.1f}% vs "
                   f"target, descent miles {descs['var_pct'].median():+.1f}%.")

    tot_stop = df["stopped_sec"].sum()
    if tot_stop > 60:
        big = md.loc[md["stop"].idxmax()]
        out.append(f"Stopped {f_time(tot_stop)} total; the largest chunk came in "
                   f"mile {int(big['mile'])} ({f_time(big['stop'])}).")

    moving_total = df["actual_delta"].sum() - tot_stop
    gaps = {g: abs(moving_total - df[f"pred_sec_{g}"].sum()) for g in GEARS}
    verdict = min(gaps, key=gaps.get)
    out.append(f"Overall, this run tracked closest to your {verdict.upper()} gear "
               f"({f_signed(moving_total - df[f'pred_sec_{verdict}'].sum())} vs its projection).")
    return out
