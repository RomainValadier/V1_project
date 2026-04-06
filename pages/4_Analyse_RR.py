from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from polar_app.clean_export import RR_CLEAN_ALGO_VERSION, export_clean_result
from polar_app.repository import ProcessedSessionRepository
from polar_app.rr_pipeline import RRCleaningParams, analyze_rr_artifacts

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
ACTIVITY_FALLBACK_LABEL = "Activite non annotee"
CORRECTION_METHODS = ["division", "fusion", "interpolation_pchip", "interpolation_lineaire"]
CORRECTION_COLORS = {
    "division": "#1d4ed8",
    "fusion": "#b45309",
    "interpolation_pchip": "#be123c",
    "interpolation_lineaire": "#7c3aed",
}


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
        .session-block-title {
            font-size: 1.08rem; font-weight: 700; color: #173427; margin: 0.35rem 0 0.15rem 0;
        }
        .session-block-meta {
            color: #55675c; font-size: 0.92rem; margin-bottom: 0.6rem;
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


def format_session_heading(session) -> str:
    annotation = session.annotation or session.session_id
    return f"{annotation} | {session.date} {session.heure_debut}"


def sort_sessions(sessions):
    return sorted(sessions, key=lambda session: (session.date, session.heure_debut, session.session_id))


def get_activity_label(session) -> str:
    activity_label = (session.activity_label or "").strip()
    return activity_label or ACTIVITY_FALLBACK_LABEL


def build_activity_session_map(sessions: list) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for session in sessions:
        grouped.setdefault(get_activity_label(session), []).append(session)
    return {label: sort_sessions(grouped[label]) for label in sorted(grouped, key=str.casefold)}


def build_session_options(sessions: list) -> tuple[list[str], dict[str, str]]:
    base_labels = [format_session_label(session) for session in sessions]
    counts = Counter(base_labels)
    labels: list[str] = []
    label_to_id: dict[str, str] = {}
    for session, base_label in zip(sessions, base_labels):
        label = f"{base_label} | {session.session_id}" if counts[base_label] > 1 else base_label
        labels.append(label)
        label_to_id[label] = session.session_id
    return labels, label_to_id


def ensure_multiselect_state(state_key: str, options: list[str], default_values: list[str]) -> None:
    current = st.session_state.get(state_key)
    valid = [value for value in current if value in options] if isinstance(current, list) else []
    if not valid:
        st.session_state[state_key] = default_values
    elif len(valid) != len(current):
        st.session_state[state_key] = valid


def ensure_selectbox_state(state_key: str, options: list[str], default_value: str) -> None:
    if st.session_state.get(state_key) not in options:
        st.session_state[state_key] = default_value


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
            out.loc[missing, "timestamp"] = start_dt + pd.to_timedelta(
                out.loc[missing, "t_offset_ms"].astype("float64"),
                unit="ms",
            )
    return out


def build_rr_clean_chart_frame(cleaned_frame: pd.DataFrame, session) -> pd.DataFrame:
    chart = ensure_timestamp(cleaned_frame, session)
    if chart.empty:
        return chart
    chart["display_rr_ms"] = chart["display_rr_ms"].astype("float64")
    chart["line_group"] = chart["display_rr_ms"].isna().astype(int).cumsum()
    chart["point_nettoye"] = chart["correction_flag"].isin(CORRECTION_METHODS)
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


def prepare_session_analysis(repository: ProcessedSessionRepository, session_id: str, params: RRCleaningParams) -> dict[str, object]:
    session, rr_frame, _ = repository.load_session_data(session_id)
    result = analyze_rr_artifacts(rr_frame, params)
    rr_chart = build_rr_clean_chart_frame(result.cleaned_frame, session)
    total_points = max(result.total_points, 1)
    non_viable_mask = (
        result.analysis_frame["run_flag"].isin(["gap_deco", "post_reconnect_deco"])
        | result.analysis_frame["label"].eq("gap_deco")
    )
    return {
        "session": session,
        "result": result,
        "clean_meta": load_clean_meta(repository, session_id),
        "rr_chart": rr_chart,
        "dense_boxes": build_dense_boxes(rr_chart, result.dense_regions_frame),
        "segments_table": build_segments_table(result, session),
        "breaks_table": build_breaks_table(result, session),
        "dense_regions_table": build_dense_regions_table(result, session),
        "label_rate_table": build_label_rate_table(result),
        "non_viable_pct": float(non_viable_mask.sum()) / float(total_points) * 100.0,
    }


def build_rr_chart_specs(session_analysis: dict[str, object]) -> tuple[alt.Chart, alt.Chart]:
    session = session_analysis["session"]
    rr_chart = session_analysis["rr_chart"]
    dense_boxes = session_analysis["dense_boxes"]
    visible_rr = rr_chart.loc[rr_chart["display_rr_ms"].notna()].copy()
    corrected_rr = visible_rr.loc[visible_rr["point_nettoye"]].copy()
    present_corrections = [
        method for method in CORRECTION_METHODS if method in corrected_rr["correction_flag"].astype(str).unique().tolist()
    ]
    correction_scale = (
        alt.Scale(domain=present_corrections, range=[CORRECTION_COLORS[method] for method in present_corrections])
        if present_corrections
        else alt.Scale(domain=["division"], range=["#1d4ed8"])
    )
    start_iso = session_start_dt(session).isoformat()
    end_iso = rr_chart["timestamp"].max().isoformat() if not rr_chart.empty and rr_chart["timestamp"].notna().any() else start_iso
    time_domain = [start_iso, end_iso]

    rr_line_chart = alt.Chart(visible_rr).mark_line(color="#2d5f43", strokeWidth=2.1).encode(
        x=alt.X("timestamp:T", title="Heure de la seance", scale=alt.Scale(domain=time_domain)),
        y=alt.Y("display_rr_ms:Q", title="RR clean (ms)"),
        detail="line_group:N",
    )
    rr_clean_plain_chart = alt.Chart(visible_rr).mark_line(color="#2d5f43", strokeWidth=2.1).encode(
        x=alt.X("timestamp:T", title="Heure de la seance", scale=alt.Scale(domain=time_domain)),
        y=alt.Y("display_rr_ms:Q", title="RR clean (ms)"),
        detail="line_group:N",
        tooltip=["timestamp:T", "display_rr_ms:Q"],
    )
    rr_corrected_chart = alt.Chart(corrected_rr).mark_point(
        filled=True,
        size=86,
        stroke="#ffffff",
        strokeWidth=1.1,
    ).encode(
        x="timestamp:T",
        y="display_rr_ms:Q",
        color=alt.Color("correction_flag:N", scale=correction_scale, title="Correction"),
        shape=alt.Shape(
            "correction_flag:N",
            scale=alt.Scale(domain=CORRECTION_METHODS, range=["circle", "square", "diamond", "triangle-up"]),
            title="Correction",
        ),
        tooltip=["timestamp:T", "display_rr_ms:Q", "label:N", "run_flag:N", "run_series_flag:N", "correction_flag:N"],
    )
    rr_dense_chart = alt.Chart(dense_boxes).mark_rect(
        stroke="#0f766e",
        strokeWidth=3.0,
        strokeDash=[8, 4],
        color="#14b8a6",
        fillOpacity=0.10,
    ).encode(
        x=alt.X("x_start:T", scale=alt.Scale(domain=time_domain)),
        x2="x_end:T",
        y="y_min:Q",
        y2="y_max:Q",
        tooltip=["dense_region_id:Q", "nb_battements:Q", "nb_artefacts:Q", "densite_artefact_pct:Q"],
    )
    annotated_chart = (rr_dense_chart + rr_line_chart + rr_corrected_chart).properties(height=380).interactive()
    plain_chart = rr_clean_plain_chart.properties(height=380).interactive()
    return annotated_chart, plain_chart


def render_rr_block(session_analysis: dict[str, object], show_heading: bool) -> None:
    session = session_analysis["session"]
    if show_heading:
        st.markdown(f'<div class="session-block-title">{format_session_heading(session)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="session-block-meta">Activite : {get_activity_label(session)}</div>', unsafe_allow_html=True)

    annotated_chart, plain_chart = build_rr_chart_specs(session_analysis)
    rr_col1, rr_col2 = st.columns(2)
    with rr_col1:
        render_section_label("RR clean annote")
        st.altair_chart(annotated_chart, use_container_width=True)
    with rr_col2:
        render_section_label("RR clean sans annotation")
        st.altair_chart(plain_chart, use_container_width=True)


def render_active_session_details(session_analysis: dict[str, object], selected_count: int) -> None:
    session = session_analysis["session"]
    result = session_analysis["result"]
    clean_meta = session_analysis["clean_meta"]

    render_section_label("Resume seance")
    active_caption = f"Seance active : {format_session_heading(session)} | " if selected_count > 1 else ""
    st.caption(
        active_caption
        + f"Version actuelle de l'algo de nettoyage : {RR_CLEAN_ALGO_VERSION} | "
        + f"Version export clean de la seance : {clean_meta.get('cleaning_algo_version', 'absente')}"
    )
    metrics = st.columns(6)
    metrics[0].metric("Segments", str(len(result.quality_segments_frame)))
    metrics[1].metric("Artefacts RR non ok", f"{result.global_non_ok_rate * 100:.2f}%")
    metrics[2].metric("RR non viables", f"{session_analysis['non_viable_pct']:.2f}%")
    metrics[3].metric("Zones denses", str(result.n_zones_denses))
    metrics[4].metric("Cassures", str(result.n_cassures_total))
    metrics[5].metric("RR clean", str(result.cleaned_points))

    table_col1, table_col2 = st.columns(2)
    with table_col1:
        render_section_label("Segments")
        st.dataframe(session_analysis["segments_table"], use_container_width=True, hide_index=True, height=320)
    with table_col2:
        render_section_label("Zones denses")
        st.dataframe(session_analysis["dense_regions_table"], use_container_width=True, hide_index=True, height=320)

    table_col3, table_col4 = st.columns(2)
    with table_col3:
        render_section_label("Cassures")
        st.dataframe(session_analysis["breaks_table"], use_container_width=True, hide_index=True, height=320)
    with table_col4:
        render_section_label("Taux d'artefact par label")
        st.dataframe(session_analysis["label_rate_table"], use_container_width=True, hide_index=True, height=320)


def main() -> None:
    st.set_page_config(page_title="Analyse RR", layout="wide", initial_sidebar_state="expanded")
    inject_styles()
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Analyse RR</div>
            <div class="hero-subtitle">Vue de synthese des RR clean, segments, zones denses, cassures et repartition des artefacts par seance.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    repository = ProcessedSessionRepository(DEFAULT_OUTPUT_DIR)
    sessions = sort_sessions(repository.list_sessions())
    if not sessions:
        st.warning("Aucune session traitee trouvee.")
        st.stop()

    params = RRCleaningParams()
    stale_sessions = sessions_requiring_recompile(repository, sessions)
    missing_count = len(stale_sessions)
    latest_session = sessions[-1]
    activity_session_map = build_activity_session_map(sessions)
    activity_options = list(activity_session_map.keys())
    default_activity = get_activity_label(latest_session)

    with st.sidebar:
        render_section_label("Selection")
        ensure_selectbox_state("analyse_rr_activity", activity_options, default_activity)
        selected_activity = st.selectbox("Activite", activity_options, key="analyse_rr_activity")

        activity_sessions = activity_session_map[selected_activity]
        activity_labels, activity_label_to_id = build_session_options(activity_sessions)
        default_selected_session_id = (
            latest_session.session_id if get_activity_label(latest_session) == selected_activity else activity_sessions[-1].session_id
        )
        default_selected_labels = [
            label for label, session_id in activity_label_to_id.items() if session_id == default_selected_session_id
        ]
        ensure_multiselect_state("analyse_rr_session_labels", activity_labels, default_selected_labels)
        selected_session_labels = st.multiselect(
            "Seances a visualiser",
            activity_labels,
            key="analyse_rr_session_labels",
            help="La multi-selection pilote uniquement l'affichage des graphiques RR.",
        )

        selected_session_ids = [activity_label_to_id[label] for label in selected_session_labels]
        selected_sessions = [session for session in activity_sessions if session.session_id in selected_session_ids]

        if selected_sessions:
            active_labels, active_label_to_id = build_session_options(selected_sessions)
            default_active_id = selected_sessions[-1].session_id
            default_active_label = next(
                label for label, session_id in active_label_to_id.items() if session_id == default_active_id
            )
            ensure_selectbox_state("analyse_rr_active_label", active_labels, default_active_label)
            active_session_label = st.selectbox(
                "Seance active pour le detail",
                active_labels,
                key="analyse_rr_active_label",
                help="Les tableaux et indicateurs detailles restent pilotes par cette seance.",
            )
            active_session_id = active_label_to_id[active_session_label]
        else:
            active_session_id = None
            st.caption("Selectionne au moins une seance pour afficher les graphiques RR.")

        if st.button(f"Compiler / recompiler les RR_clean a mettre a jour ({missing_count})", use_container_width=True):
            exported, failed = compile_missing_clean(repository, sessions, params)
            if exported:
                st.success(f"RR_clean generes pour {len(exported)} seance(s).")
            if failed:
                st.error("Echec sur : " + ", ".join(failed))
            st.rerun()

    if not selected_session_ids:
        st.info("Aucune seance selectionnee pour cette activite.")
        st.stop()

    session_analyses = {
        session_id: prepare_session_analysis(repository, session_id, params)
        for session_id in selected_session_ids
    }

    render_section_label("RR clean multi-seances" if len(selected_session_ids) > 1 else "RR clean")
    for index, session_id in enumerate(selected_session_ids):
        render_rr_block(session_analyses[session_id], show_heading=len(selected_session_ids) > 1)
        if len(selected_session_ids) > 1 and index < len(selected_session_ids) - 1:
            st.markdown("---")

    render_active_session_details(session_analyses[active_session_id], len(selected_session_ids))


if __name__ == "__main__":
    main()
