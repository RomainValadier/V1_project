from __future__ import annotations

import json
import os
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from polar_app.clean_export import RR_CLEAN_ALGO_VERSION, export_clean_result
from polar_app.repository import ProcessedSessionRepository
from polar_app.rr_pipeline import RRCleaningParams, analyze_rr_artifacts

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(214, 154, 48, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(25, 102, 78, 0.12), transparent 28%),
                linear-gradient(180deg, #f7f3e8 0%, #fbfaf6 48%, #edf5f0 100%);
        }
        .block-container { max-width: 1650px; padding-top: 1.35rem; }
        .hero-card {
            padding: 1.05rem 1.25rem; border-radius: 20px; margin-bottom: 1rem;
            background: linear-gradient(135deg, rgba(255,255,255,0.97), rgba(242,247,243,0.95));
            border: 1px solid rgba(33, 72, 52, 0.10); box-shadow: 0 14px 36px rgba(38, 61, 47, 0.08);
        }
        .hero-title { font-size: 1.9rem; font-weight: 760; color: #173427; }
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
    clean_state = "clean disponible" if session.rr_clean_filepath else "clean absent"
    return f"{annotation} | {session.date} {session.heure_debut} | {clean_state}"


def sort_sessions(sessions):
    return sorted(sessions, key=lambda session: (session.date, session.heure_debut, session.session_id))


def session_start_dt(session) -> datetime:
    if session.fc_start_ts:
        return datetime.fromisoformat(session.fc_start_ts)
    return datetime.fromisoformat(f"{session.date}T{session.heure_debut}")


def ensure_timestamp(frame: pd.DataFrame, session) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out
    start_dt = session_start_dt(session)
    if "timestamp" not in out.columns:
        out["timestamp"] = pd.NaT
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if "t_offset_ms" in out.columns:
        missing = out["timestamp"].isna()
        if missing.any():
            out.loc[missing, "timestamp"] = start_dt + pd.to_timedelta(out.loc[missing, "t_offset_ms"].astype("float64"), unit="ms")
    return out


def build_rr_clean_chart_frame(cleaned_frame: pd.DataFrame, session) -> pd.DataFrame:
    chart = ensure_timestamp(cleaned_frame, session)
    if chart.empty:
        return chart
    chart["display_rr_ms"] = chart["display_rr_ms"].astype("float64")
    chart["line_group"] = chart["display_rr_ms"].isna().astype(int).cumsum()
    chart["point_nettoye"] = chart["correction_flag"].isin(["division", "fusion", "interpolation_pchip", "interpolation_lineaire"])
    return chart


def build_dense_boxes(rr_chart: pd.DataFrame, dense_regions_frame: pd.DataFrame) -> pd.DataFrame:
    if rr_chart.empty or dense_regions_frame.empty:
        return pd.DataFrame()
    visible = rr_chart.loc[rr_chart["display_rr_ms"].notna()].copy()
    if visible.empty:
        return pd.DataFrame()
    y_min_all = float(visible["display_rr_ms"].min()) - 45.0
    y_max_all = float(visible["display_rr_ms"].max()) + 45.0
    boxes: list[dict[str, object]] = []
    for row in dense_regions_frame.itertuples(index=False):
        overlap = rr_chart.loc[
            (rr_chart["source_index_start"].fillna(-1).astype(int) <= int(row.index_fin))
            & (rr_chart["source_index_end"].fillna(-1).astype(int) >= int(row.index_debut))
            & rr_chart["display_rr_ms"].notna()
        ].copy()
        if overlap.empty:
            continue
        x_start = overlap["timestamp"].min()
        x_end = overlap["timestamp"].max()
        if pd.isna(x_start) or pd.isna(x_end):
            continue
        if x_end <= x_start:
            x_end = x_start + pd.Timedelta(seconds=1)
        boxes.append(
            {
                "dense_region_id": int(row.dense_region_id),
                "x_start": x_start,
                "x_end": x_end,
                "y_min": y_min_all,
                "y_max": y_max_all,
                "nb_battements": int(row.nb_battements),
                "nb_artefacts": int(row.nb_artefacts),
                "densite_artefact_pct": float(row.densite_artefact_pct),
            }
        )
    return pd.DataFrame(boxes)


def build_segments_table(result, session) -> pd.DataFrame:
    segments = result.quality_segments_frame.copy()
    if segments.empty:
        return segments
    analysis = ensure_timestamp(result.analysis_frame, session)
    starts = analysis["timestamp"]
    rows: list[dict[str, object]] = []
    for row in segments.itertuples(index=False):
        ts_start = starts.iloc[int(row.index_debut)] if int(row.index_debut) < len(starts) else pd.NaT
        ts_end = starts.iloc[int(row.index_fin)] if int(row.index_fin) < len(starts) else pd.NaT
        rows.append(
            {
                "segment_final_id": int(row.segment_final_id),
                "timestamp_debut": ts_start,
                "timestamp_fin": ts_end,
                "duree_min": round(float(row.duree_ms) / 60000.0, 2),
                "taux_artefact_pct": round(float(row.taux_artefact or 0.0) * 100.0, 2),
                "status": row.qualite_segment,
            }
        )
    return pd.DataFrame(rows)


def build_breaks_table(result, session) -> pd.DataFrame:
    breaks = result.breaks_frame.copy()
    if breaks.empty:
        return breaks
    analysis = ensure_timestamp(result.analysis_frame, session)
    rows: list[dict[str, object]] = []
    for row in breaks.itertuples(index=False):
        idx_start = int(row.index_debut)
        idx_end = int(row.index_fin)
        ts_start = analysis.iloc[idx_start]["timestamp"] if idx_start < len(analysis) else pd.NaT
        ts_end = analysis.iloc[idx_end]["timestamp"] if idx_end < len(analysis) else pd.NaT
        rows.append(
            {
                "raison": row.flag_source,
                "timestamp_debut": ts_start,
                "timestamp_fin": ts_end,
                "longueur_battements": int(row.longueur_battements),
                "duree_s": round(float(row.duree_ms) / 1000.0, 2),
            }
        )
    return pd.DataFrame(rows)


def build_dense_regions_table(result, session) -> pd.DataFrame:
    dense = result.dense_regions_frame.copy()
    if dense.empty:
        return dense
    analysis = ensure_timestamp(result.analysis_frame, session)
    rows: list[dict[str, object]] = []
    for row in dense.itertuples(index=False):
        idx_start = int(row.index_debut)
        idx_end = int(row.index_fin)
        ts_start = analysis.iloc[idx_start]["timestamp"] if idx_start < len(analysis) else pd.NaT
        ts_end = analysis.iloc[idx_end]["timestamp"] if idx_end < len(analysis) else pd.NaT
        rows.append(
            {
                "dense_region_id": int(row.dense_region_id),
                "timestamp_debut": ts_start,
                "timestamp_fin": ts_end,
                "longueur_battements": int(row.nb_battements),
                "pourcentage_artefact": float(row.densite_artefact_pct),
            }
        )
    return pd.DataFrame(rows)


def build_label_rate_table(result) -> pd.DataFrame:
    analysis = result.analysis_frame.copy()
    if analysis.empty:
        return pd.DataFrame()
    total = float(len(analysis))
    labels = analysis["label"].astype(str)
    artifact_labels = labels[labels.ne("ok")]
    rows = []
    for label, count in artifact_labels.value_counts().to_dict().items():
        rows.append(
            {
                "label": label,
                "nombre": int(count),
                "taux_pct_sur_total": round((float(count) / total) * 100.0, 2),
            }
        )
    return pd.DataFrame(rows)


def load_clean_meta(repository: ProcessedSessionRepository, session_id: str) -> dict:
    paths = repository.get_clean_paths(session_id)
    meta_path = paths["clean_meta_absolute"]
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path, encoding="utf-8-sig") as file_obj:
        return json.load(file_obj)


def sessions_requiring_recompile(repository: ProcessedSessionRepository, sessions: list) -> list:
    stale_sessions = []
    for session in sessions:
        if not repository.has_clean_export(session.session_id):
            stale_sessions.append(session)
            continue
        clean_meta = load_clean_meta(repository, session.session_id)
        if clean_meta.get("cleaning_algo_version") != RR_CLEAN_ALGO_VERSION:
            stale_sessions.append(session)
    return stale_sessions


def compile_missing_clean(repository: ProcessedSessionRepository, sessions: list, params: RRCleaningParams) -> tuple[list[str], list[str]]:
    exported: list[str] = []
    failed: list[str] = []
    target_sessions = sessions_requiring_recompile(repository, sessions)
    progress = st.progress(0.0, text="Compilation / recompilation des RR_clean...")
    status = st.empty()
    total = max(len(target_sessions), 1)
    for idx, session in enumerate(target_sessions, start=1):
        status.info(f"Nettoyage en cours : {session.annotation or session.session_id} ({idx}/{len(target_sessions)})")
        try:
            session_obj, rr_frame, _ = repository.load_session_data(session.session_id)
            result = analyze_rr_artifacts(rr_frame, params)
            export_clean_result(repository, session_obj, result, params)
            exported.append(session.session_id)
        except Exception:
            failed.append(session.session_id)
        progress.progress(idx / total, text=f"Compilation / recompilation des RR_clean... {idx}/{len(target_sessions)}")
    progress.empty()
    status.empty()
    return exported, failed


def main() -> None:
    st.set_page_config(page_title="Analyse RR", layout="wide", initial_sidebar_state="expanded")
    inject_styles()
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Analyse RR</div>
            <div class="hero-subtitle">Vue de synth?se des RR clean, segments, zones denses, cassures et r?partition des artefacts par s?ance.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    repository = ProcessedSessionRepository(DEFAULT_OUTPUT_DIR)
    sessions = sort_sessions(repository.list_sessions())
    if not sessions:
        st.warning("Aucune session trait?e trouv?e.")
        st.stop()

    params = RRCleaningParams()
    stale_sessions = sessions_requiring_recompile(repository, sessions)
    missing_count = len(stale_sessions)

    with st.sidebar:
        render_section_label("S?lection")
        session_options = {format_session_label(session): session.session_id for session in sessions}
        selected_label = st.selectbox("Session", list(session_options.keys()), index=len(session_options) - 1)
        if st.button(f"Compiler / recompiler les RR_clean ? mettre ? jour ({missing_count})", use_container_width=True):
            exported, failed = compile_missing_clean(repository, sessions, params)
            if exported:
                st.success(f"RR_clean g?n?r?s pour {len(exported)} s?ance(s).")
            if failed:
                st.error("?chec sur : " + ", ".join(failed))
            st.rerun()

    session_id = session_options[selected_label]
    session, rr_frame, _ = repository.load_session_data(session_id)
    result = analyze_rr_artifacts(rr_frame, params)
    clean_meta = load_clean_meta(repository, session_id)
    rr_chart = build_rr_clean_chart_frame(result.cleaned_frame, session)
    dense_boxes = build_dense_boxes(rr_chart, result.dense_regions_frame)
    segments_table = build_segments_table(result, session)
    breaks_table = build_breaks_table(result, session)
    dense_regions_table = build_dense_regions_table(result, session)
    label_rate_table = build_label_rate_table(result)

    total_points = max(result.total_points, 1)
    non_viable_mask = (
        result.analysis_frame["run_flag"].isin(["gap_deco", "post_reconnect_deco"])
        | result.analysis_frame["label"].eq("gap_deco")
    )
    non_viable_pct = float(non_viable_mask.sum()) / float(total_points) * 100.0

    render_section_label("R?sum? s?ance")
    st.caption(f"Version actuelle de l'algo de nettoyage : {RR_CLEAN_ALGO_VERSION} | Version export clean de la s?ance : {clean_meta.get('cleaning_algo_version', 'absente')}")
    metrics = st.columns(6)
    metrics[0].metric("Segments", str(len(result.quality_segments_frame)))
    metrics[1].metric("Artefacts RR non ok", f"{result.global_non_ok_rate * 100:.2f}%")
    metrics[2].metric("RR non viables", f"{non_viable_pct:.2f}%")
    metrics[3].metric("Zones denses", str(result.n_zones_denses))
    metrics[4].metric("Cassures", str(result.n_cassures_total))
    metrics[5].metric("RR clean", str(result.cleaned_points))

    render_section_label("RR clean")
    visible_rr = rr_chart.loc[rr_chart["display_rr_ms"].notna()].copy()
    corrected_rr = visible_rr.loc[visible_rr["point_nettoye"]].copy()
    correction_colors = {
        "division": "#1d4ed8",
        "fusion": "#b45309",
        "interpolation_pchip": "#be123c",
        "interpolation_lineaire": "#7c3aed",
    }
    present_corrections = [
        method for method in ["division", "fusion", "interpolation_pchip", "interpolation_lineaire"]
        if method in corrected_rr["correction_flag"].astype(str).unique().tolist()
    ]
    correction_scale = alt.Scale(
        domain=present_corrections,
        range=[correction_colors[method] for method in present_corrections],
    ) if present_corrections else alt.Scale(domain=["division"], range=["#1d4ed8"])
    time_domain = [session_start_dt(session).isoformat(), rr_chart["timestamp"].max().isoformat() if not rr_chart.empty and rr_chart["timestamp"].notna().any() else session_start_dt(session).isoformat()]

    rr_line_chart = alt.Chart(visible_rr).mark_line(color="#2d5f43", strokeWidth=2.1).encode(
        x=alt.X("timestamp:T", title="Heure de la s?ance", scale=alt.Scale(domain=time_domain)),
        y=alt.Y("display_rr_ms:Q", title="RR clean (ms)"),
        detail="line_group:N",
    )
    rr_clean_plain_chart = alt.Chart(visible_rr).mark_line(color="#2d5f43", strokeWidth=2.1).encode(
        x=alt.X("timestamp:T", title="Heure de la s?ance", scale=alt.Scale(domain=time_domain)),
        y=alt.Y("display_rr_ms:Q", title="RR clean (ms)"),
        detail="line_group:N",
        tooltip=["timestamp:T", "display_rr_ms:Q"],
    )
    rr_corrected_chart = alt.Chart(corrected_rr).mark_point(filled=True, size=86, stroke="#ffffff", strokeWidth=1.1).encode(
        x="timestamp:T",
        y="display_rr_ms:Q",
        color=alt.Color("correction_flag:N", scale=correction_scale, title="Correction"),
        shape=alt.Shape(
            "correction_flag:N",
            scale=alt.Scale(domain=["division", "fusion", "interpolation_pchip", "interpolation_lineaire"], range=["circle", "square", "diamond", "triangle-up"]),
            title="Correction",
        ),
        tooltip=["timestamp:T", "display_rr_ms:Q", "label:N", "run_flag:N", "run_series_flag:N", "correction_flag:N"],
    )
    rr_dense_chart = alt.Chart(dense_boxes).mark_rect(stroke="#0f766e", strokeWidth=3.0, strokeDash=[8, 4], color="#14b8a6", fillOpacity=0.10).encode(
        x=alt.X("x_start:T", scale=alt.Scale(domain=time_domain)),
        x2="x_end:T",
        y="y_min:Q",
        y2="y_max:Q",
        tooltip=["dense_region_id:Q", "nb_battements:Q", "nb_artefacts:Q", "densite_artefact_pct:Q"],
    )
    rr_col1, rr_col2 = st.columns(2)
    with rr_col1:
        render_section_label("RR clean annot?")
        st.altair_chart((rr_dense_chart + rr_line_chart + rr_corrected_chart).properties(height=380).interactive(), use_container_width=True)
    with rr_col2:
        render_section_label("RR clean sans annotation")
        st.altair_chart(rr_clean_plain_chart.properties(height=380).interactive(), use_container_width=True)

    table_col1, table_col2 = st.columns(2)
    with table_col1:
        render_section_label("Segments")
        st.dataframe(segments_table, use_container_width=True, hide_index=True, height=320)
    with table_col2:
        render_section_label("Zones denses")
        st.dataframe(dense_regions_table, use_container_width=True, hide_index=True, height=320)

    table_col3, table_col4 = st.columns(2)
    with table_col3:
        render_section_label("Cassures")
        st.dataframe(breaks_table, use_container_width=True, hide_index=True, height=320)
    with table_col4:
        render_section_label("Taux d'artefact par label")
        st.dataframe(label_rate_table, use_container_width=True, hide_index=True, height=320)


if __name__ == "__main__":
    main()
