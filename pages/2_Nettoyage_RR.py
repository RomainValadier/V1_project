from __future__ import annotations

import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from polar_app.clean_export import export_clean_result
from polar_app.repository import ProcessedSessionRepository
from polar_app.rr_pipeline import (
    CORRECTION_FLAG_ORDER,
    DECO_FLAG_ORDER,
    LABEL_ORDER,
    RAW_ARTIFACT_COLORS,
    RUN_FLAG_ORDER,
    RRCleaningParams,
    analyze_rr_artifacts,
)

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(218, 165, 32, 0.10), transparent 30%),
                radial-gradient(circle at top right, rgba(46, 139, 87, 0.10), transparent 28%),
                linear-gradient(180deg, #f6f3ea 0%, #fbfaf7 52%, #f3f6f1 100%);
        }
        .block-container { max-width: 1600px; padding-top: 1.4rem; }
        .hero-card {
            padding: 1rem 1.2rem; border-radius: 18px; margin-bottom: 1rem;
            background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(246,248,243,0.94));
            border: 1px solid rgba(46, 84, 61, 0.10); box-shadow: 0 12px 36px rgba(41, 60, 47, 0.08);
        }
        .hero-title { font-size: 1.85rem; font-weight: 700; color: #163325; }
        .hero-subtitle { color: #526357; font-size: 0.98rem; }
        .section-chip {
            display: inline-block; padding: 0.3rem 0.7rem; border-radius: 999px;
            background: rgba(55, 110, 79, 0.10); color: #2d5f43; font-size: 0.8rem; font-weight: 600;
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
    return f"{annotation} | {session.date} {session.heure_debut}"


def sort_sessions(sessions):
    return sorted(sessions, key=lambda session: (session.date, session.heure_debut, session.session_id))


def build_parameter_inputs(defaults: RRCleaningParams) -> RRCleaningParams:
    render_section_label("Paramètres du pipeline v3")
    values: dict[str, float | int] = {}
    with st.expander("Bornes physiologiques", expanded=True):
        col1, col2 = st.columns(2)
        values["rr_min_ms"] = col1.number_input("RR_MIN_MS", 100, 2000, defaults.rr_min_ms, 10)
        values["rr_max_ms"] = col2.number_input("RR_MAX_MS", 200, 4000, defaults.rr_max_ms, 10)
    with st.expander("Détection différentielle", expanded=True):
        cols = st.columns(4)
        values["alpha"] = cols[0].number_input("ALPHA", 0.1, 20.0, defaults.alpha, 0.1, format="%.2f")
        values["window_th"] = cols[0].number_input("WINDOW_TH", 5, 301, defaults.window_th, 2)
        values["window_median"] = cols[1].number_input("WINDOW_MEDIAN", 3, 101, defaults.window_median, 2)
        values["threshold_drr"] = cols[1].number_input("THRESHOLD_DRR", 0.1, 10.0, defaults.threshold_drr, 0.1, format="%.2f")
        values["threshold_mrr"] = cols[2].number_input("THRESHOLD_MRR", 0.1, 20.0, defaults.threshold_mrr, 0.1, format="%.2f")
    with st.expander("Runs, densité et interpolation", expanded=False):
        cols = st.columns(4)
        values["seuil_run_court_max"] = cols[0].number_input("SEUIL_RUN_COURT_MAX", 1, 20, defaults.seuil_run_court_max, 1)
        values["seuil_run_moyen_max"] = cols[0].number_input("SEUIL_RUN_MOYEN_MAX", 2, 30, defaults.seuil_run_moyen_max, 1)
        values["seuil_deco_interpoler"] = cols[1].number_input("SEUIL_DECO_INTERPOLER", 1, 30, defaults.seuil_deco_interpoler, 1)
        values["window_densite"] = cols[1].number_input("WINDOW_DENSITE", 5, 100, defaults.window_densite, 1)
        values["seuil_densite"] = cols[2].number_input("SEUIL_DENSITE", 0.05, 1.0, defaults.seuil_densite, 0.05, format="%.2f")
        values["y_court"] = cols[2].number_input("Y_COURT", 2, 10, defaults.y_court, 1)
        values["tolerance_court"] = cols[3].number_input("TOLERANCE_COURT", 0, 5, defaults.tolerance_court, 1)
        values["y_moyen"] = cols[3].number_input("Y_MOYEN", 3, 15, defaults.y_moyen, 1)
    with st.expander("Cassures et qualité", expanded=False):
        cols = st.columns(4)
        values["tolerance_moyen"] = cols[0].number_input("TOLERANCE_MOYEN", 0, 6, defaults.tolerance_moyen, 1)
        values["chauffe_gap_court"] = cols[0].number_input("CHAUFFE_GAP_COURT", 1, 12, defaults.chauffe_gap_court, 1)
        values["chauffe_gap_long"] = cols[1].number_input("CHAUFFE_GAP_LONG", 1, 12, defaults.chauffe_gap_long, 1)
        values["seuil_gap_duree_ms"] = cols[1].number_input("SEUIL_GAP_DUREE_MS", 1000, 60000, defaults.seuil_gap_duree_ms, 1000)
        values["seuil_qualite_court"] = cols[2].number_input("SEUIL_QUALITE_COURT", 0.0, 1.0, defaults.seuil_qualite_court, 0.01, format="%.2f")
        values["seuil_qualite_long"] = cols[2].number_input("SEUIL_QUALITE_LONG", 0.0, 1.0, defaults.seuil_qualite_long, 0.01, format="%.2f")
        values["duree_seuil_segment_ms"] = cols[3].number_input("DUREE_SEUIL_SEGMENT_MS", 10000, 600000, defaults.duree_seuil_segment_ms, 1000)
    return RRCleaningParams(**{key: (int(value) if isinstance(getattr(defaults, key), int) else float(value)) for key, value in values.items()})


def build_raw_chart(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    chart = frame.copy()
    chart["t_min"] = chart["t_offset_ms"] / 60000.0
    chart["rr_plot_ms"] = chart["rr_interval_ms"].where(chart["rr_interval_ms"] > 0, np.nan)
    chart["line_group"] = chart["rr_plot_ms"].isna().astype(int).cumsum()
    artifacts = chart.loc[chart["label"].isin(list(RAW_ARTIFACT_COLORS.keys()))].copy()
    return chart.loc[chart["rr_plot_ms"].notna()].copy(), artifacts


def build_clean_chart(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chart = frame.copy()
    if chart.empty:
        return chart, chart, chart
    chart["timeline_step_ms"] = chart["timeline_step_ms"].fillna(0.0)
    chart["t_min"] = chart["timeline_step_ms"].cumsum() / 60000.0
    chart["line_group"] = chart["display_rr_ms"].isna().astype(int).cumsum()
    line_data = chart.loc[chart["display_rr_ms"].notna()].copy()
    corrected = chart.loc[chart["correction_flag"].isin(["division", "fusion", "interpolation_cubique", "interpolation_lineaire"]) & chart["display_rr_ms"].notna()].copy()
    excluded = chart.loc[chart["display_rr_ms"].isna()].copy()
    return line_data, corrected, excluded


def build_dense_boxes(cleaned_frame: pd.DataFrame) -> pd.DataFrame:
    if cleaned_frame.empty or "dense_region_id" not in cleaned_frame.columns:
        return pd.DataFrame()
    dense = cleaned_frame.loc[cleaned_frame["dense_region_id"].gt(0)].copy()
    if dense.empty:
        return pd.DataFrame()
    dense["t_min"] = dense["timeline_step_ms"].fillna(0.0).cumsum() / 60000.0
    boxes = dense.groupby("dense_region_id", as_index=False).agg(x_start=("t_min", "min"), x_end=("t_min", "max"), y_min=("display_rr_ms", "min"), y_max=("display_rr_ms", "max"), nb_points=("dense_region_id", "size"))
    boxes["x_start"] -= 0.03
    boxes["x_end"] += 0.03
    boxes["y_min"] -= 45.0
    boxes["y_max"] += 45.0
    return boxes


def counts_frame(items: dict[str, int], label: str) -> pd.DataFrame:
    return pd.DataFrame([{label: key, "nombre": value} for key, value in items.items()])

def main() -> None:
    st.set_page_config(page_title="Nettoyage RR", layout="wide", initial_sidebar_state="expanded")
    inject_styles()
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Nettoyage des artefacts RR</div>
            <div class="hero-subtitle">Pipeline RR v3 avec détection des déconnexions importées, exclusion segmentaire, attribution des runs, correction et qualification FC / HRR / RMSSD.</div>
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

    params = build_parameter_inputs(RRCleaningParams())
    session, rr_frame, _ = repository.load_session_data(session_options[selected_label])
    result = analyze_rr_artifacts(rr_frame, params)
    clean_paths = export_clean_result(repository, session, result, params)

    st.caption(f"Export clean automatique : {clean_paths['rr_clean_relative']} | {clean_paths['fc_clean_relative']}")

    render_section_label("Résumé")
    row1 = st.columns(6)
    row2 = st.columns(6)
    row1[0].metric("RR bruts", str(result.total_points))
    row1[1].metric("RR clean", str(result.cleaned_points))
    row1[2].metric("RR exploitables FC", str(result.ok_rr_total))
    row1[3].metric("RR non viables", str(result.non_viable_rr_total))
    row1[4].metric("Cassures déco", str(result.n_cassures_deco))
    row1[5].metric("Cassures artefact", str(result.n_cassures_artefact))
    row2[0].metric("Cassures total", str(result.n_cassures_total))
    row2[1].metric("Zones denses", str(result.n_zones_denses))
    row2[2].metric("Points zones denses", str(result.dense_points_count))
    row2[3].metric("Segments actifs", str(result.n_segments_actifs))
    row2[4].metric("Segments exclus", str(result.n_segments_exclus))
    row2[5].metric("Qualité globale", result.global_quality_label)
    st.caption(f"Taux non-ok global : {result.global_non_ok_rate * 100:.2f}% | Taux de correction hors cassures : {result.correction_rate_outside_breaks * 100:.2f}%")

    counts_col1, counts_col2, counts_col3, counts_col4 = st.columns(4)
    with counts_col1:
        render_section_label("LABELS")
        st.dataframe(counts_frame(result.label_counts, "label"), use_container_width=True, hide_index=True)
    with counts_col2:
        render_section_label("RUN_FLAGS")
        st.dataframe(counts_frame(result.run_flag_counts, "run_flag"), use_container_width=True, hide_index=True)
    with counts_col3:
        render_section_label("DECO_FLAGS")
        st.dataframe(counts_frame(result.deco_flag_counts, "deco_flag"), use_container_width=True, hide_index=True)
    with counts_col4:
        render_section_label("CORRECTION_FLAGS")
        st.dataframe(counts_frame(result.correction_counts, "correction_flag"), use_container_width=True, hide_index=True)

    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        render_section_label("Segments initiaux")
        st.dataframe(result.segments_frame, use_container_width=True, hide_index=True)
    with info_col2:
        render_section_label("Qualité des segments")
        st.dataframe(result.quality_segments_frame, use_container_width=True, hide_index=True)
    with info_col3:
        render_section_label("Cassures")
        if result.breaks_frame.empty:
            st.info("Aucune cassure détectée.")
        else:
            st.dataframe(result.breaks_frame, use_container_width=True, hide_index=True)

    raw_line, raw_artifacts = build_raw_chart(result.analysis_frame)
    clean_line, clean_corrected, clean_excluded = build_clean_chart(result.cleaned_frame)
    dense_boxes = build_dense_boxes(result.cleaned_frame)

    artifact_scale = alt.Scale(domain=list(RAW_ARTIFACT_COLORS.keys()), range=list(RAW_ARTIFACT_COLORS.values()))
    raw_line_chart = alt.Chart(raw_line).mark_line(color="#94a3b8", strokeWidth=1.5).encode(
        x=alt.X("t_min:Q", title="Temps (min)"),
        y=alt.Y("rr_plot_ms:Q", title="RR brut (ms)"),
        detail="line_group:N",
    )
    raw_artifact_chart = alt.Chart(raw_artifacts).mark_circle(size=60).encode(
        x="t_min:Q",
        y="rr_plot_ms:Q",
        color=alt.Color("label:N", scale=artifact_scale, title="LABEL"),
        tooltip=["t_min:Q", "rr_interval_ms:Q", "label:N", "run_flag:N", "deco_flag:N"],
    )
    clean_line_chart = alt.Chart(clean_line).mark_line(color="#2d5f43", strokeWidth=2.1).encode(
        x=alt.X("t_min:Q", title="Temps cumulé clean (min)"),
        y=alt.Y("display_rr_ms:Q", title="RR clean (ms)"),
        detail="line_group:N",
    )
    correction_scale = alt.Scale(
        domain=["division", "fusion", "interpolation_cubique", "interpolation_lineaire"],
        range=["#1d4ed8", "#b45309", "#c2410c", "#7c3aed"],
    )
    clean_corrected_chart = alt.Chart(clean_corrected).mark_circle(size=70, stroke="#ffffff", strokeWidth=1.1).encode(
        x="t_min:Q",
        y="display_rr_ms:Q",
        color=alt.Color("correction_flag:N", scale=correction_scale, title="Correction"),
        tooltip=["t_min:Q", "rr_interval_ms:Q", "label:N", "run_flag:N", "correction_flag:N", "fc_ok:N", "hrr_ok:N", "rmssd_ok:N"],
    )
    clean_excluded_chart = alt.Chart(clean_excluded).mark_tick(thickness=2, size=18, color="#475569").encode(
        x="t_min:Q",
        tooltip=["t_min:Q", "label:N", "run_flag:N", "correction_flag:N", "segment_status:N"],
    )
    dense_outline = alt.Chart(dense_boxes).mark_rect(stroke="#0f766e", strokeWidth=3.0, strokeDash=[8, 4], color="#14b8a6", fillOpacity=0.10).encode(
        x="x_start:Q", x2="x_end:Q", y="y_min:Q", y2="y_max:Q", tooltip=["dense_region_id:Q", "nb_points:Q"]
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        render_section_label("RR bruts")
        st.altair_chart((raw_line_chart + raw_artifact_chart).properties(height=320).interactive(), use_container_width=True)
        st.caption("Les artefacts bruts sont colorés selon leur LABEL v3.")
    with chart_col2:
        render_section_label("RR clean")
        st.altair_chart((dense_outline + clean_line_chart + clean_corrected_chart + clean_excluded_chart).properties(height=320).interactive(), use_container_width=True)
        st.caption("Les zones denses run_serie sont encadr?es en vert d'eau et les points effectivement nettoy?s sont color?s selon la m?thode de correction.")

    render_section_label("Variables intermédiaires")
    variable_columns = [
        "t_offset_ms", "rr_interval_ms", "source_pipeline_flag", "deco_flag", "label_initial", "label", "run_flag",
        "dRR_ms", "Th1_ms", "dRR_norm", "med_locale_ms", "mRR_brut_ms", "mRR_ms", "Th2_ms", "mRR_norm", "S21", "S22",
        "initial_segment_id", "initial_segment_status", "segment_final_id", "dense_region_id", "correction_flag_raw",
    ]
    available = [column for column in variable_columns if column in result.analysis_frame.columns]
    st.dataframe(result.analysis_frame[available], use_container_width=True, height=320)

    table_col1, table_col2 = st.columns(2)
    with table_col1:
        render_section_label("Table RR bruts")
        raw_columns = ["t_offset_ms", "rr_interval_ms", "source_pipeline_flag", "deco_flag", "label", "run_flag", "initial_segment_id", "initial_segment_status", "segment_final_id", "dense_region_id", "correction_flag_raw"]
        raw_table = result.analysis_frame[raw_columns]
        st.dataframe(raw_table, use_container_width=True, height=340)
    with table_col2:
        render_section_label("Table RR clean")
        clean_table_source = result.cleaned_frame.assign(
            point_nettoye=result.cleaned_frame["correction_flag"].isin(["division", "fusion", "interpolation_cubique", "interpolation_lineaire"]),
            zone_dense_artefact=result.cleaned_frame["dense_region_id"].fillna(0).gt(0),
        )
        clean_columns = ["cleaned_index", "rr_interval_ms", "label", "run_flag", "deco_flag", "correction_flag", "point_nettoye", "zone_dense_artefact", "fc_ok", "hrr_ok", "rmssd_ok", "segment_status", "quality_segment_label", "dense_region_id", "source_reference"]
        cleaned_table = clean_table_source[clean_columns] if show_full_tables else clean_table_source[clean_columns].head(700)
        st.dataframe(cleaned_table, use_container_width=True, height=340)



if __name__ == "__main__":
    main()
