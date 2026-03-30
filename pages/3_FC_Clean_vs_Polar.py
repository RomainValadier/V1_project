from __future__ import annotations

import os
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from polar_app.clean_export import FC_CLEAN_MIN_VIABLE_POINTS, FC_CLEAN_WINDOW_BEATS
from polar_app.repository import ProcessedSessionRepository

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(214, 154, 48, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(36, 122, 88, 0.12), transparent 28%),
                linear-gradient(180deg, #f5f0e5 0%, #fbfaf6 48%, #eef5ef 100%);
        }
        .block-container { max-width: 1600px; padding-top: 1.4rem; }
        .hero-card {
            padding: 1.1rem 1.3rem; border-radius: 20px; margin-bottom: 1rem;
            background: linear-gradient(135deg, rgba(255,255,255,0.97), rgba(244,248,241,0.95));
            border: 1px solid rgba(35, 73, 52, 0.10); box-shadow: 0 14px 36px rgba(38, 61, 47, 0.08);
        }
        .hero-title { font-size: 1.9rem; font-weight: 750; color: #173427; }
        .hero-subtitle { color: #55675c; font-size: 1rem; }
        .section-chip {
            display: inline-block; padding: 0.3rem 0.72rem; border-radius: 999px;
            background: rgba(47, 104, 74, 0.10); color: #2a5c40; font-size: 0.8rem; font-weight: 650;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_label(label: str) -> None:
    st.markdown(f'<div class="section-chip">{label}</div>', unsafe_allow_html=True)


def format_session_label(session) -> str:
    annotation = session.annotation or session.session_id
    clean_state = "clean disponible" if session.rr_clean_filepath and session.fc_clean_filepath else "clean absent"
    return f"{annotation} | {session.date} {session.heure_debut} | {clean_state}"


def sort_sessions(sessions):
    return sorted(sessions, key=lambda session: (session.date, session.heure_debut, session.session_id))


def session_start_dt(session) -> datetime:
    if session.fc_start_ts:
        return datetime.fromisoformat(session.fc_start_ts)
    return datetime.fromisoformat(f"{session.date}T{session.heure_debut}")


def ensure_rr_clean_columns(rr_clean_frame: pd.DataFrame) -> pd.DataFrame:
    frame = rr_clean_frame.copy()
    if frame.empty:
        return frame
    if "fc_ok" not in frame.columns:
        if "rr_viability_label" in frame.columns:
            frame["fc_ok"] = frame["rr_viability_label"].eq("RR_OK")
        elif "analytic_use" in frame.columns:
            analytic = frame["analytic_use"].astype(str).str.upper()
            frame["fc_ok"] = analytic.str.contains("FC") | analytic.eq("TRUE")
        else:
            frame["fc_ok"] = frame["rr_interval_ms"].notna()
    if "hrr_ok" not in frame.columns:
        frame["hrr_ok"] = frame["fc_ok"]
    if "rmssd_ok" not in frame.columns:
        if "rr_viability_label" in frame.columns:
            frame["rmssd_ok"] = frame["rr_viability_label"].eq("RR_OK")
        else:
            frame["rmssd_ok"] = False
    for column, default in {
        "label": "ok",
        "run_flag": "ok",
        "deco_flag": "ok",
        "correction_flag": "ok",
    }.items():
        if column not in frame.columns:
            frame[column] = default
    return frame


def ensure_fc_clean_columns(fc_clean_frame: pd.DataFrame) -> pd.DataFrame:
    frame = fc_clean_frame.copy()
    if frame.empty:
        return frame
    for column, default in {
        "window_viable_points": pd.NA,
        "window_size_beats": pd.NA,
        "center_fc_ok": False,
        "center_hrr_ok": False,
        "center_rmssd_ok": False,
    }.items():
        if column not in frame.columns:
            frame[column] = default
    return frame


def build_rr_clean_chart_frame(rr_clean_frame: pd.DataFrame) -> pd.DataFrame:
    chart = rr_clean_frame.copy()
    if chart.empty:
        return chart
    chart["timestamp"] = pd.to_datetime(chart["timestamp"])
    chart["display_rr_ms"] = np.where(chart["fc_ok"].eq(True), chart["rr_interval_ms"], np.nan)
    chart["line_group"] = chart["display_rr_ms"].isna().astype(int).cumsum()
    return chart


def build_fc_clean_chart_frame(fc_clean_frame: pd.DataFrame) -> pd.DataFrame:
    chart = fc_clean_frame.copy()
    if chart.empty:
        return chart
    chart["timestamp"] = pd.to_datetime(chart["timestamp"])
    chart["line_group"] = chart["bpm_clean"].isna().astype(int).cumsum()
    chart["series"] = "FC clean"
    chart["bpm"] = chart["bpm_clean"]
    return chart


def build_polar_fc_chart_frame(hr_frame: pd.DataFrame) -> pd.DataFrame:
    chart = hr_frame.copy()
    if chart.empty:
        return pd.DataFrame(columns=["timestamp", "bpm", "line_group", "series"])
    chart["timestamp"] = pd.to_datetime(chart["timestamp"])
    chart["bpm"] = chart["bpm"].where(chart["bpm"] > 0, np.nan)
    break_mask = chart["bpm"].isna()
    if "segment_id" in chart.columns:
        break_mask = break_mask | chart["segment_id"].ne(chart["segment_id"].shift(1)).fillna(False)
    break_mask = break_mask | chart["timestamp"].diff().gt(pd.Timedelta(seconds=3)).fillna(False)
    chart["line_group"] = break_mask.astype(int).cumsum()
    chart["series"] = "FC Polar"
    return chart[["timestamp", "bpm", "line_group", "series"]].copy()


def build_time_domain(session, rr_clean_frame: pd.DataFrame, fc_clean_frame: pd.DataFrame, hr_frame: pd.DataFrame) -> list[str]:
    start = session_start_dt(session)
    max_candidates = [start]
    for frame in (rr_clean_frame, fc_clean_frame, hr_frame):
        if not frame.empty and "timestamp" in frame.columns:
            timestamps = pd.to_datetime(frame["timestamp"])
            if not timestamps.empty:
                max_candidates.append(timestamps.max().to_pydatetime())
    return [start.isoformat(), max(max_candidates).isoformat()]


def main() -> None:
    st.set_page_config(page_title="FC Clean vs Polar", layout="wide", initial_sidebar_state="expanded")
    inject_styles()
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Comparaison FC clean et FC Polar</div>
            <div class="hero-subtitle">Visualisation des RR clean v3, de la FC recalculée sur fenêtre centrée de 5 battements, et de la FC issue de Polar sur un axe horaire absolu.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    repository = ProcessedSessionRepository(DEFAULT_OUTPUT_DIR)
    sessions = sort_sessions(repository.list_sessions())
    if not sessions:
        st.warning("Aucune session traitée trouvée.")
        st.stop()

    with st.sidebar:
        render_section_label("Sélection")
        session_options = {format_session_label(session): session.session_id for session in sessions}
        selected_label = st.selectbox("Session", list(session_options.keys()), index=len(session_options) - 1)
        show_full_tables = st.toggle("Afficher les tables complètes", value=False)

    session_id = session_options[selected_label]
    session, _, hr_frame = repository.load_session_data(session_id)
    if not repository.has_clean_export(session_id):
        st.warning("Cette séance n'a pas encore d'export clean. Ouvre d'abord la page Nettoyage RR pour générer les fichiers clean.")
        st.stop()

    session, rr_clean_frame, fc_clean_frame, clean_meta = repository.load_clean_data(session_id)
    rr_clean_frame = ensure_rr_clean_columns(rr_clean_frame)
    fc_clean_frame = ensure_fc_clean_columns(fc_clean_frame)
    rr_chart = build_rr_clean_chart_frame(rr_clean_frame)
    clean_fc_chart = build_fc_clean_chart_frame(fc_clean_frame)
    polar_fc_chart = build_polar_fc_chart_frame(hr_frame)
    time_domain = build_time_domain(session, rr_chart, clean_fc_chart, polar_fc_chart)

    rr_clean_visible = int(rr_chart["display_rr_ms"].notna().sum()) if not rr_chart.empty else 0
    fc_clean_points = int(clean_fc_chart["bpm"].notna().sum()) if not clean_fc_chart.empty else 0
    ok_rr_total = int(rr_clean_frame["fc_ok"].sum()) if not rr_clean_frame.empty else 0
    non_viable_rr_total = int((~rr_clean_frame[["fc_ok", "hrr_ok", "rmssd_ok"]].any(axis=1)).sum()) if not rr_clean_frame.empty else 0

    render_section_label("Résumé séance")
    metrics = st.columns(6)
    metrics[0].metric("RR bruts", str(session.nb_battements_rr))
    metrics[1].metric("RR clean visibles", str(rr_clean_visible))
    metrics[2].metric("RR FC exploitables", str(ok_rr_total))
    metrics[3].metric("RR non viables", str(non_viable_rr_total))
    metrics[4].metric("Points FC clean", str(fc_clean_points))
    metrics[5].metric("Qualité globale", str(clean_meta.get("global_quality_label", "-")))

    render_section_label("Paramètres FC clean")
    st.caption(
        f"Fenêtre centrée de {clean_meta.get('fc_clean_window_beats', FC_CLEAN_WINDOW_BEATS)} battements | "
        f"minimum {clean_meta.get('fc_clean_min_viable_points', FC_CLEAN_MIN_VIABLE_POINTS)} RR exploitables FC | "
        f"export du {clean_meta.get('export_timestamp', '-') }"
    )

    fc_combined = pd.concat([
        clean_fc_chart[["timestamp", "bpm", "line_group", "series"]] if not clean_fc_chart.empty else pd.DataFrame(columns=["timestamp", "bpm", "line_group", "series"]),
        polar_fc_chart[["timestamp", "bpm", "line_group", "series"]] if not polar_fc_chart.empty else pd.DataFrame(columns=["timestamp", "bpm", "line_group", "series"]),
    ], ignore_index=True)

    color_scale = alt.Scale(domain=["FC clean", "FC Polar"], range=["#1f6f50", "#b45309"])
    fc_chart = alt.Chart(fc_combined.dropna(subset=["bpm"])).mark_line(strokeWidth=2.4).encode(
        x=alt.X("timestamp:T", title="Heure de la séance", scale=alt.Scale(domain=time_domain)),
        y=alt.Y("bpm:Q", title="FC (bpm)"),
        color=alt.Color("series:N", scale=color_scale, title="Série"),
        detail="line_group:N",
        tooltip=["timestamp:T", "series:N", "bpm:Q"],
    )

    rr_chart_alt = alt.Chart(rr_chart.dropna(subset=["display_rr_ms"])).mark_line(color="#2d5f43", strokeWidth=2.2).encode(
        x=alt.X("timestamp:T", title="Heure de la séance", scale=alt.Scale(domain=time_domain)),
        y=alt.Y("display_rr_ms:Q", title="RR clean (ms)"),
        detail="line_group:N",
        tooltip=["timestamp:T", "display_rr_ms:Q", "label:N", "run_flag:N", "correction_flag:N"],
    )

    col1, col2 = st.columns([1.45, 1.0])
    with col1:
        render_section_label("FC clean vs FC Polar")
        st.altair_chart(fc_chart.properties(height=360), use_container_width=True)
        st.caption("La FC clean est calculée sur une fenêtre centrée de 5 battements. Si moins de 2 RR exploitables FC sont présents dans la fenêtre, la courbe laisse un trou.")
    with col2:
        render_section_label("RR clean")
        st.altair_chart(rr_chart_alt.properties(height=360), use_container_width=True)
        st.caption("Les points non exploitables restent traçables dans la table mais ne sont pas tracés sur cette courbe clean.")

    table_col1, table_col2 = st.columns(2)
    with table_col1:
        render_section_label("Table RR clean")
        rr_columns = ["timestamp", "rr_interval_ms", "display_rr_ms", "label", "run_flag", "deco_flag", "correction_flag", "fc_ok", "hrr_ok", "rmssd_ok"]
        rr_table = rr_clean_frame[rr_columns] if show_full_tables else rr_clean_frame[rr_columns].head(500)
        st.dataframe(rr_table, use_container_width=True, height=340, hide_index=True)
    with table_col2:
        render_section_label("Table FC clean")
        fc_columns = ["timestamp", "bpm_clean", "window_viable_points", "window_size_beats", "center_fc_ok", "center_hrr_ok", "center_rmssd_ok"]
        fc_table = fc_clean_frame[fc_columns] if show_full_tables else fc_clean_frame[fc_columns].head(500)
        st.dataframe(fc_table, use_container_width=True, height=340, hide_index=True)


if __name__ == "__main__":
    main()
