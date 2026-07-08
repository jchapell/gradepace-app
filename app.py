import datetime
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import gradepace as gp
from github_store import GitHubStore

# =========================================================================
# GRADEPACE STRATEGY ENGINE — deployed app (phase 1: single athlete)
# Engine logic lives in gradepace.py (identical to the Colab notebook's).
# Persistent cache lives in a private GitHub repo via github_store.py.
# =========================================================================

st.set_page_config(page_title="GradePace Planner", page_icon="🏔️", layout="wide")

REQUIRED_SECRETS = [
    "STRAVA_CLIENT_ID", "STRAVA_CLIENT_SECRET", "STRAVA_REFRESH_TOKEN",
    "GITHUB_TOKEN", "GITHUB_DATA_REPO",
]
missing = [k for k in REQUIRED_SECRETS if k not in st.secrets]
if missing:
    st.error(f"Missing secrets: {', '.join(missing)}. "
             "Add them in the app's Settings → Secrets (see README).")
    st.stop()

store = GitHubStore(st.secrets["GITHUB_DATA_REPO"], st.secrets["GITHUB_TOKEN"])


# =========================================================================
# STRAVA
# =========================================================================
def strava_auth():
    """Silent auth via refresh token; returns (headers, athlete_id, athlete_name)."""
    res = requests.post("https://www.strava.com/api/v3/oauth/token", data={
        "client_id": st.secrets["STRAVA_CLIENT_ID"],
        "client_secret": st.secrets["STRAVA_CLIENT_SECRET"],
        "grant_type": "refresh_token",
        "refresh_token": st.secrets["STRAVA_REFRESH_TOKEN"],
    }, timeout=30).json()
    token = res.get("access_token")
    if not token:
        raise RuntimeError(f"Strava auth failed: {res}")
    headers = {"Authorization": f"Bearer {token}"}
    ath = requests.get("https://www.strava.com/api/v3/athlete", headers=headers, timeout=30).json()
    return headers, ath["id"], f"{ath.get('firstname','')} {ath.get('lastname','')}".strip()


def rate_limited_get(url, headers, params, status):
    for attempt in range(20):
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code != 429:
            return r
        status.write(f"⏳ Strava rate limit — waiting 60s (attempt {attempt + 1})...")
        time.sleep(60)
    return r


def sync_strava(start_date, end_date, activity_types):
    """Incremental harvest: fetch only uncached runs, then push cache to GitHub."""
    headers = st.session_state["strava_headers"]
    athlete_id = st.session_state["athlete_id"]
    streams, meta = st.session_state["streams"], st.session_state["meta"]

    status = st.status("Listing activities...", expanded=True)
    after = int(datetime.datetime.combine(start_date, datetime.time.min).timestamp())
    before = int(datetime.datetime.combine(end_date, datetime.time.max).timestamp())
    acts, page = [], 1
    while True:
        r = rate_limited_get("https://www.strava.com/api/v3/athlete/activities",
                             headers, {"per_page": 100, "page": page,
                                       "after": after, "before": before}, status).json()
        if not r:
            break
        acts += [a for a in r if a.get("type") in activity_types]
        page += 1
        time.sleep(0.2)

    to_fetch = gp.missing_ids(acts, meta)
    status.update(label=f"{len(acts)} runs in window — {len(to_fetch)} new to fetch "
                        f"({len(acts) - len(to_fetch)} already cached)", state="running")

    new_streams, new_meta = [], []
    progress = st.progress(0.0) if to_fetch else None
    for idx, act in enumerate(to_fetch):
        r = rate_limited_get(f"https://www.strava.com/api/v3/activities/{act['id']}/streams",
                             headers, {"keys": "time,distance,altitude,heartrate",
                                       "key_by_type": "true"}, status)
        if r.status_code == 200:
            d = r.json()
            t = d.get("time", {}).get("data", [])
            dist = d.get("distance", {}).get("data", [])
            alt = d.get("altitude", {}).get("data", [])
            hr = d.get("heartrate", {}).get("data", [])
            if len(t) == len(dist) == len(alt) and len(t) > 1:
                has_hr = len(hr) == len(t)
                for i in range(len(t)):
                    new_streams.append(dict(activity_id=act["id"], time_sec=t[i],
                                            distance_meters=dist[i], altitude_meters=alt[i],
                                            heart_rate_bpm=hr[i] if has_hr else np.nan))
                new_meta.append(dict(activity_id=act["id"], name=act["name"],
                                     type=act["type"], start_date=act["start_date"]))
        progress.progress((idx + 1) / len(to_fetch),
                          text=f"{idx + 1}/{len(to_fetch)}: {act['name'][:40]}")
        time.sleep(0.15)

    streams, meta = gp.merge_new(streams, meta, new_streams, new_meta)
    st.session_state["streams"], st.session_state["meta"] = streams, meta

    if new_meta:
        status.update(label="Saving cache to GitHub...", state="running")
        store.save_athlete_cache(athlete_id, streams, meta)
    status.update(label=f"Sync complete: {len(new_meta)} new runs added "
                        f"(cache: {meta.shape[0]} activities)", state="complete", expanded=False)


# =========================================================================
# SESSION BOOTSTRAP (auth once, pull cache from GitHub once)
# =========================================================================
if "athlete_id" not in st.session_state:
    with st.spinner("Connecting to Strava and loading your cache from GitHub..."):
        try:
            headers, athlete_id, name = strava_auth()
        except Exception as e:
            st.error(f"Strava authentication failed: {e}")
            st.stop()
        st.session_state["strava_headers"] = headers
        st.session_state["athlete_id"] = athlete_id
        st.session_state["athlete_name"] = name
        streams, meta = store.load_athlete_cache(athlete_id, gp.STREAM_COLS, gp.META_COLS)
        st.session_state["streams"], st.session_state["meta"] = streams, meta

streams = st.session_state["streams"]
meta = st.session_state["meta"]

# =========================================================================
# SIDEBAR
# =========================================================================
st.sidebar.markdown(f"**Athlete:** {st.session_state['athlete_name']}")
st.sidebar.caption(f"Cache: {meta.shape[0]} activities / {len(streams):,} points")

st.sidebar.header("1 · Data window")
today = datetime.date.today()
col_a, col_b = st.sidebar.columns(2)
start_date = col_a.date_input("From", value=today - datetime.timedelta(days=365), max_value=today)
end_date = col_b.date_input("To", value=today, max_value=today)
activity_types = st.sidebar.multiselect("Activity types", ["Trail Run", "Run"],
                                        default=["Trail Run", "Run"])
if st.sidebar.button("🔄 Sync Strava (incremental)", type="primary",
                     help="Fetches only runs the cache has never seen, then saves to GitHub."):
    sync_strava(start_date, end_date, activity_types)
    st.rerun()

st.sidebar.header("2 · Model dials")
fatigue_rate = st.sidebar.slider("Fatigue rate (%/mile)", 0.0, 3.0, 1.0, 0.1)
fatigue_onset = st.sidebar.number_input("Fatigue onset (mile)", 0.0, 200.0, 1.0, 0.5,
                                        help="Fatigue decay starts accumulating after this mile.")
altitude_rate = st.sidebar.slider("Altitude drag (%/1000ft)", 0.0, 5.0, 2.0, 0.1)
planned_stops_min = st.sidebar.number_input("Planned stops (minutes)", 0, 600, 0, 5,
                                            help="Aid stations, summits, photos. Added to the adjusted "
                                                 "projection to give Projected Elapsed Time.")
allow_benefit = st.sidebar.checkbox("Below Baseline Elev. Speedup", value=False,
                                    help="If checked, courses below your training altitude run faster.")

with st.sidebar.expander("Import existing cache (one-time migration)"):
    st.caption("Upload the two parquet files from your Google Drive GradePace folder.")
    up_s = st.file_uploader("gradepace_streams.parquet", type=["parquet"], key="mig_s")
    up_m = st.file_uploader("gradepace_activities.parquet", type=["parquet"], key="mig_m")
    if up_s and up_m and st.button("Import & save to GitHub"):
        with st.spinner("Merging and uploading..."):
            imp_s, imp_m = pd.read_parquet(up_s), pd.read_parquet(up_m)
            merged_m = pd.concat([meta, imp_m]).drop_duplicates(subset="activity_id", keep="first")
            known = set(meta["activity_id"].astype("int64")) if len(meta) else set()
            new_ids = set(merged_m["activity_id"].astype("int64")) - known
            merged_s = pd.concat([streams, imp_s[imp_s["activity_id"].astype("int64").isin(new_ids)]],
                                 ignore_index=True)
            st.session_state["streams"], st.session_state["meta"] = merged_s, merged_m
            store.save_athlete_cache(st.session_state["athlete_id"], merged_s, merged_m)
        st.success(f"Imported. Cache now holds {merged_m.shape[0]} activities.")
        st.rerun()

# =========================================================================
# MAIN
# =========================================================================
st.title("🏔️ GradePace Planner")

window_df, n_runs = gp.select_range(streams, meta, start_date, end_date, activity_types) \
    if len(meta) else (pd.DataFrame(columns=gp.STREAM_COLS), 0)

if n_runs == 0:
    st.info("No cached runs in this window yet. Hit **Sync Strava** in the sidebar "
            "(or import your existing cache under the sidebar expander).")
    st.stop()

profile, baseline_ft = gp.build_profile(window_df)

with st.expander(f"🏔️ Pacing profile — {n_runs} runs | baseline {baseline_ft:,.0f} ft", expanded=True):
    pfig = go.Figure()
    grade_labels = [b.split("(")[1].rstrip(")") for b in profile["grade_band"]]
    pfig.add_bar(
        x=grade_labels, y=profile["pace_median"], marker_color="#5B8FF9",
        customdata=np.stack([profile["grade_band"],
                             [gp.f_pace(v) for v in profile["pace_median"]],
                             profile["n_samples"]], axis=-1),
        hovertemplate="%{customdata[0]}<br>Median: %{customdata[1]} /mi"
                      "<br>Samples: %{customdata[2]:,}<extra></extra>")
    pfig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                       yaxis=dict(title="Median pace (min/mi)"),
                       xaxis=dict(title="Grade"))
    st.plotly_chart(pfig, use_container_width=True)
    disp = profile.copy()
    disp["Pace F / M / S"] = disp.apply(
        lambda r: f"{gp.f_pace(r.pace_fast)} / {gp.f_pace(r.pace_median)} / {gp.f_pace(r.pace_slow)}", axis=1)
    disp["HR F / M / S"] = disp.apply(
        lambda r: (f"{r.hr_fast:.0f} / {r.hr_median:.0f} / {r.hr_slow:.0f}"
                   if pd.notna(r.hr_median) else "N/A"), axis=1)
    st.caption("F = your faster-quartile / M = typical / S = slower-quartile pace in each "
               "grade band, from all pace samples across these runs.")
    st.dataframe(disp[["grade_band", "Pace F / M / S", "HR F / M / S", "n_samples"]],
                 use_container_width=True, hide_index=True)

st.header("Course engine")
st.caption("Course GPX → pacing plan. Watch GPX (with timestamps) → plan + variance + calibration.")
gpx_file = st.file_uploader("Upload GPX", type=["gpx"])
if gpx_file is None:
    st.stop()

gpx_df = gp.parse_gpx(gpx_file)
df = gp.simulate(gpx_df, profile, baseline_ft,
                 fatigue_rate=fatigue_rate, altitude_rate=altitude_rate,
                 allow_altitude_benefit=allow_benefit, fatigue_onset_mile=fatigue_onset)
has_watch = bool(df["has_telemetry"].iloc[0])
total_mi = df["delta_dist_miles"].sum()
gain = df[df["delta_ele"] > 0]["delta_ele"].sum() * gp.M_TO_FT
loss = abs(df[df["delta_ele"] < 0]["delta_ele"].sum() * gp.M_TO_FT)

mode = "📊 Post-run analysis (telemetry detected)" if has_watch else "🔮 Prediction (no telemetry)"
st.subheader(mode)
st.caption(f"{gpx_file.name} | {total_mi:.2f} mi | +{gain:,.0f}/-{loss:,.0f} ft ({gain - loss:+,.0f}) | "
           f"fatigue {fatigue_rate}%/mi after mile {fatigue_onset:g} | "
           f"altitude {altitude_rate}%/1000ft above {baseline_ft:,.0f} ft")

# --- raw vs adjusted projections (+ actuals) ---
r1 = st.columns(4)
r1[0].markdown("**Raw Projected Moving Time**")
r1[1].metric("Fast", gp.f_time(df["raw_sec_fast"].sum()))
r1[2].metric("Median", gp.f_time(df["raw_sec_median"].sum()))
r1[3].metric("Slow", gp.f_time(df["raw_sec_slow"].sum()))
r2 = st.columns(4)
r2[0].markdown("**Adjusted Projected Moving Time** *(+fatigue & altitude)*")
r2[1].metric("Fast", gp.f_time(df["pred_sec_fast"].sum()),
             delta=gp.f_signed(df["pred_sec_fast"].sum() - df["raw_sec_fast"].sum()), delta_color="inverse")
r2[2].metric("Median", gp.f_time(df["pred_sec_median"].sum()),
             delta=gp.f_signed(df["pred_sec_median"].sum() - df["raw_sec_median"].sum()), delta_color="inverse")
r2[3].metric("Slow", gp.f_time(df["pred_sec_slow"].sum()),
             delta=gp.f_signed(df["pred_sec_slow"].sum() - df["raw_sec_slow"].sum()), delta_color="inverse")
if planned_stops_min > 0:
    stops_sec = planned_stops_min * 60.0
    r3 = st.columns(4)
    r3[0].markdown(f"**Projected elapsed** *(+ {planned_stops_min} min stops)*")
    r3[1].metric("Fast", gp.f_time(df["pred_sec_fast"].sum() + stops_sec))
    r3[2].metric("Median", gp.f_time(df["pred_sec_median"].sum() + stops_sec))
    r3[3].metric("Slow", gp.f_time(df["pred_sec_slow"].sum() + stops_sec))

# 5. Actuals in a visually distinct bordered block
if has_watch:
    total_actual = df["actual_delta"].sum()
    stopped = df["stopped_sec"].sum()
    with st.container(border=True):
        st.markdown("**:orange[⌚ Actuals — from your watch]**")
        a = st.columns(4)
        a[0].metric("Watch total", gp.f_time(total_actual))
        a[1].metric("Moving", gp.f_time(total_actual - stopped))
        a[2].metric("Stopped", gp.f_time(stopped))
        a[3].metric("vs Adjusted Median", gp.f_signed(total_actual - df["pred_sec_median"].sum()))

# --- band time budget ---
st.markdown("**⏱️ Time by grade band (adjusted)**")
band_rows = []
for band, g in df.groupby("grade_band"):
    mi = g["delta_dist_miles"].sum()
    band_rows.append({
        "Grade band": band,
        "Fast": gp.f_time(g["pred_sec_fast"].sum()),
        "Median": gp.f_time(g["pred_sec_median"].sum()),
        "Slow": gp.f_time(g["pred_sec_slow"].sum()),
        "Actual*": gp.f_time(g["actual_delta"].sum()) if has_watch else "N/A",
        "Share": f"{mi / total_mi * 100:.0f}% · {mi:.2f} mi",
    })
st.dataframe(pd.DataFrame(band_rows), use_container_width=True, hide_index=True)
if has_watch:
    st.caption("*Actual includes stopped time within each band.")

# --- unified mile table ---
st.markdown("**📊 Mile-by-mile** — targets are altitude+fatigue adjusted")
mile_rows = []
for m, g in df.groupby("mile_bucket"):
    mi = g["delta_dist_miles"].sum()
    if mi < 0.1:
        continue
    climb = g[g["delta_ele"] > 0]["delta_ele"].sum() * gp.M_TO_FT
    drop = abs(g[g["delta_ele"] < 0]["delta_ele"].sum() * gp.M_TO_FT)
    if has_watch:
        act = g["actual_delta"].sum()
        stop = g["stopped_sec"].sum()
        moving = gp.f_pace(((act - stop) / 60.0) / mi)
        var = gp.f_signed((act - stop) - g["pred_sec_median"].sum())
        stop_s = gp.f_time(stop)
    else:
        moving, var, stop_s = "N/A", "N/A", "N/A"
    mile_rows.append({
        "Mile": int(m),
        "Target (F/M/S)": f"{gp.f_pace(g['sim_pace_fast'].mean())} / "
                          f"{gp.f_pace(g['sim_pace_median'].mean())} / "
                          f"{gp.f_pace(g['sim_pace_slow'].mean())}",
        "Split (M)": gp.f_time(g["pred_sec_median"].sum()),
        "Fatigue": gp.f_signed(g["fatigue_cost_median"].sum()),
        "Alt Cost": gp.f_signed(g["altitude_cost_median"].sum()),
        "Moving": moving,
        "Var": var,
        "Stop": stop_s,
        "Vert": f"+{climb:.0f}/-{drop:.0f} ({climb - drop:+.0f})",
    })
st.dataframe(pd.DataFrame(mile_rows), use_container_width=True, hide_index=True,
             height=min(38 * (len(mile_rows) + 1), 600))

# --- pace bars + cumulative time lines ---
miles, tgt_pace, act_pace, stop_pmi, stop_fmt, cum_tgt, cum_act = [], [], [], [], [], [], []
run_tgt, run_act = 0.0, 0.0
for m, g in df.groupby("mile_bucket"):
    mi = g["delta_dist_miles"].sum()
    if mi < 0.1:
        continue
    miles.append(int(m))
    tgt_pace.append(g["sim_pace_median"].mean())
    run_tgt += g["pred_sec_median"].sum()
    cum_tgt.append(run_tgt)
    if has_watch:
        act = g["actual_delta"].sum()
        stop = g["stopped_sec"].sum()
        act_pace.append(((act - stop) / 60.0) / mi)
        stop_pmi.append((stop / 60.0) / mi)
        stop_fmt.append(gp.f_time(stop))
        run_act += act
        cum_act.append(run_act)

fig = go.Figure()
fig.add_bar(x=miles, y=tgt_pace, name="Target pace (M)", marker_color="#5B8FF9",
            offsetgroup=0,
            customdata=[gp.f_pace(v) for v in tgt_pace],
            hovertemplate="Mile %{x}<br>Target: %{customdata} /mi<extra></extra>")
if has_watch:
    fig.add_bar(x=miles, y=act_pace, name="Actual moving pace", marker_color="#F6903D",
                offsetgroup=1,
                customdata=[gp.f_pace(v) for v in act_pace],
                hovertemplate="Mile %{x}<br>Moving: %{customdata} /mi<extra></extra>")
    fig.add_bar(x=miles, y=stop_pmi, base=act_pace, name="Stopped time",
                marker_color="#1a1a1a", offsetgroup=1,
                customdata=stop_fmt,
                hovertemplate="Mile %{x}<br>Stopped: %{customdata}<extra></extra>")
fig.add_scatter(x=miles, y=[s / 3600.0 for s in cum_tgt], yaxis="y2", mode="lines",
                name="Cumulative target", line=dict(color="#1A56B0", width=3),
                customdata=[gp.f_time(s) for s in cum_tgt],
                hovertemplate="Mile %{x}<br>Cum target: %{customdata}<extra></extra>")
if has_watch:
    fig.add_scatter(x=miles, y=[s / 3600.0 for s in cum_act], yaxis="y2", mode="lines",
                    name="Cumulative actual (elapsed)", line=dict(color="#C2570C", width=3, dash="dot"),
                    customdata=[gp.f_time(s) for s in cum_act],
                    hovertemplate="Mile %{x}<br>Cum elapsed: %{customdata}<extra></extra>")
fig.update_layout(
    barmode="group", height=430,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    xaxis=dict(title="Mile", dtick=1 if len(miles) <= 35 else 5),
    yaxis=dict(title="Pace (min/mi)"),
    yaxis2=dict(title="Cumulative time (hr)", overlaying="y", side="right", showgrid=False),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# --- calibration insights ---
if has_watch:
    st.markdown("**🎯 Calibration insights**")
    for line in gp.calibration_insights(df, fatigue_rate, altitude_rate):
        st.markdown(f"- {line}")
else:
    st.caption("🎯 After the race, upload your watch GPX here for variance + calibration analysis.")
