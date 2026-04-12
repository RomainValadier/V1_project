from __future__ import annotations

import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from polar_app.clean_export import export_clean_result
from polar_app.etape_2bis import (
    LABEL_2BIS_A,
    LABEL_2BIS_B,
    LABEL_2BIS_NONE,
    run_etape_2bis,
)
from polar_app.iterative_lipponen import PASS_MARKERS, iterative_reclassification
from polar_app.repository import ProcessedSessionRepository
from polar_app.rr_pipeline import (
    CORRECTION_FLAG_ORDER,
    DECO_FLAG_ORDER,
    LABEL_ARTEFACT_ABSOLU,
    LABEL_COURT,
    LABEL_FAUX_BATTEMENT,
    LABEL_GAP_DECO,
    LABEL_LONG,
    LABEL_MANQUE,
    LABEL_OK,
    LABEL_ORDER,
    RAW_ARTIFACT_COLORS,
    RUN_FLAG_ORDER,
    RRCleaningParams,
    analyze_rr_artifacts,
)

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_2BIS_PARAMS = {
    "ratio_max": 1.25,
    "k_ref": 15,
    "n_ok_max": 8,
}
DEFAULT_ITERATIVE_PARAMS = {
    "max_iter": 3,
    "dense_threshold": 0.15,
    "dense_window": 121,
    "qd_floor": 5.0,
    "k_drr_max": 5,
    "seuil_rendement": 3,
}
RAW_LABEL_COLORS = {
    LABEL_OK: "#64748b",
    LABEL_GAP_DECO: "#cbd5e1",
    LABEL_ARTEFACT_ABSOLU: RAW_ARTIFACT_COLORS[LABEL_ARTEFACT_ABSOLU],
    LABEL_MANQUE: RAW_ARTIFACT_COLORS[LABEL_MANQUE],
    LABEL_FAUX_BATTEMENT: RAW_ARTIFACT_COLORS[LABEL_FAUX_BATTEMENT],
    LABEL_LONG: RAW_ARTIFACT_COLORS[LABEL_LONG],
    LABEL_COURT: RAW_ARTIFACT_COLORS[LABEL_COURT],
}
LABEL_2BIS_SHAPES = {
    LABEL_2BIS_NONE: "circle",
    LABEL_2BIS_A: "diamond",
    LABEL_2BIS_B: "triangle-up",
}
NON_OK_LABEL_ORDER = [label for label in LABEL_ORDER if label != LABEL_OK]
RAW_2BIS_COLOR_SCALE = alt.Scale(domain=[LABEL_2BIS_A, LABEL_2BIS_B], range=["#8b5cf6", "#f472b6"])
RAW_2BIS_SHAPE_SCALE = alt.Scale(domain=[LABEL_2BIS_A, LABEL_2BIS_B], range=["diamond", "triangle-up"])
ITERATIVE_ANALYSIS_COLUMNS = [
    "label_iteratif",
    "pass_detected",
    "zone_dense_artefact",
    "mrr_local",
    "qd_drr_local",
    "qd_mrr_local",
    "th1_local",
    "th2_local",
    "drr_iter",
    "drr_gap",
    "suspect_zone_dense",
]
ITERATIVE_PASS_DOMAIN = [f"Pass {pass_number}" for pass_number in sorted(PASS_MARKERS.keys())]
ITERATIVE_PASS_COLORS = [PASS_MARKERS[pass_number][0] for pass_number in sorted(PASS_MARKERS.keys())]
ITERATIVE_PASS_SHAPES = [PASS_MARKERS[pass_number][1] for pass_number in sorted(PASS_MARKERS.keys())]


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


def build_2bis_inputs() -> tuple[bool, dict[str, float | int]]:
    with st.expander("?? ?tape 2 bis ? Post-classification (exp?rimental)", expanded=False):
        activer_2bis = st.toggle("Activer la passe A", value=False)
        cols = st.columns(2)
        ratio_max = cols[0].number_input(
            "Plafond physiologique (ratio_max)",
            min_value=1.10,
            max_value=1.30,
            value=float(DEFAULT_2BIS_PARAMS["ratio_max"]),
            step=0.01,
            format="%.2f",
        )
        k_ref = cols[0].number_input(
            "Beats r?f?rence pr?-bosse (k_ref)",
            min_value=5,
            max_value=30,
            value=int(DEFAULT_2BIS_PARAMS["k_ref"]),
            step=1,
        )
        n_ok_max = cols[1].number_input(
            "Beats ok max examin?s passe A",
            min_value=1,
            max_value=15,
            value=int(DEFAULT_2BIS_PARAMS["n_ok_max"]),
            step=1,
        )
    return activer_2bis, {
        "ratio_max": float(ratio_max),
        "k_ref": int(k_ref),
        "n_ok_max": int(n_ok_max),
    }


def build_iterative_inputs() -> tuple[bool, dict[str, float | int]]:
    activer_iteratif = st.sidebar.toggle(
        "?? Nettoyage it?ratif Lipponen",
        value=False,
        help="Recalcule les seuils Lipponen dans les zones denses en artefacts avant une correction unique finale.",
    )
    with st.sidebar.expander("?? Param?tres it?ration", expanded=False):
        max_iter = st.slider("Nombre max d'it?rations", 2, 5, int(DEFAULT_ITERATIVE_PARAMS["max_iter"]))
        dense_threshold = st.slider("Seuil zone dense (%)", 5, 30, int(DEFAULT_ITERATIVE_PARAMS["dense_threshold"] * 100), step=1) / 100.0
        dense_window = st.slider("Fen?tre zone dense (battements)", 91, 181, int(DEFAULT_ITERATIVE_PARAMS["dense_window"]), step=10)
        qd_floor = st.number_input("Plancher QD (ms)", min_value=1.0, max_value=20.0, value=float(DEFAULT_ITERATIVE_PARAMS["qd_floor"]), step=0.5, format="%.1f")
        k_drr_max = st.slider("K_DRR_MAX (distance max pr?d?cesseur ok)", 3, 10, int(DEFAULT_ITERATIVE_PARAMS["k_drr_max"]))
        seuil_rendement = st.slider("Seuil rendement d?croissant", 1, 10, int(DEFAULT_ITERATIVE_PARAMS["seuil_rendement"]))
    return activer_iteratif, {
        "max_iter": int(max_iter),
        "dense_threshold": float(dense_threshold),
        "dense_window": int(dense_window),
        "qd_floor": float(qd_floor),
        "k_drr_max": int(k_drr_max),
        "seuil_rendement": int(seuil_rendement),
    }


def _map_rr_clean_from_cleaned(cleaned_frame: pd.DataFrame, n_points: int) -> np.ndarray:
    rr_clean = np.full(n_points, np.nan)
    if cleaned_frame.empty:
        return rr_clean
    grouped = cleaned_frame.groupby(["source_index_start", "source_index_end"], sort=False, dropna=False)
    for (start, end), group in grouped:
        if pd.isna(start) or pd.isna(end):
            continue
        start_i = int(start)
        end_i = int(end)
        values = group["rr_interval_ms"].astype(float).to_numpy()
        if start_i == end_i:
            rr_clean[start_i] = float(values[0]) if len(values) else np.nan
            continue
        targets = np.arange(start_i, end_i + 1, dtype=float)
        if len(values) == len(targets):
            rr_clean[start_i : end_i + 1] = values
            continue
        source_positions = np.linspace(float(start_i), float(end_i), num=max(len(values), 1))
        rr_clean[start_i : end_i + 1] = np.interp(targets, source_positions, values) if len(values) else np.nan
    return rr_clean


def _map_clean_text_from_cleaned(cleaned_frame: pd.DataFrame, n_points: int, column: str, default: str) -> pd.Series:
    mapped = np.full(n_points, default, dtype=object)
    if cleaned_frame.empty or column not in cleaned_frame.columns:
        return pd.Series(mapped)
    grouped = cleaned_frame.groupby(["source_index_start", "source_index_end"], sort=False, dropna=False)
    for (start, end), group in grouped:
        if pd.isna(start) or pd.isna(end):
            continue
        start_i = int(start)
        end_i = int(end)
        value = str(group.iloc[0][column]) if pd.notna(group.iloc[0][column]) else default
        mapped[start_i : end_i + 1] = value
    return pd.Series(mapped)


def _aggregate_iterative_label(values: list[str]) -> str:
    non_default = [value for value in values if value and value != "aucun"]
    if not non_default:
        return "aucun"
    unique = list(dict.fromkeys(non_default))
    return unique[0] if len(unique) == 1 else "mixte"


def attach_iterative_overlay(result, iterative_result):
    analysis = result.analysis_frame.copy()
    iterative_analysis = iterative_result.result.analysis_frame.copy()
    for column in ITERATIVE_ANALYSIS_COLUMNS:
        if column in iterative_analysis.columns:
            analysis[column] = iterative_analysis[column].values
    analysis["correction_flag_final"] = _map_clean_text_from_cleaned(result.cleaned_frame, len(analysis), "correction_flag", "ok").astype(str)
    analysis["rr_clean_ms"] = _map_rr_clean_from_cleaned(result.cleaned_frame, len(analysis))
    result.analysis_frame = analysis

    cleaned = result.cleaned_frame.copy()
    if cleaned.empty:
        result.cleaned_frame = cleaned
        return result

    def aggregate_label(row):
        start = int(row["source_index_start"]) if pd.notna(row.get("source_index_start")) else None
        end = int(row["source_index_end"]) if pd.notna(row.get("source_index_end")) else None
        if start is None or end is None:
            return "aucun"
        values = analysis.loc[start:end, "label_iteratif"].fillna("aucun").astype(str).tolist() if "label_iteratif" in analysis.columns else ["aucun"]
        return _aggregate_iterative_label(values)

    def aggregate_max(row, column: str, default: int = 0) -> int:
        start = int(row["source_index_start"]) if pd.notna(row.get("source_index_start")) else None
        end = int(row["source_index_end"]) if pd.notna(row.get("source_index_end")) else None
        if start is None or end is None or column not in analysis.columns:
            return default
        return int(pd.to_numeric(analysis.loc[start:end, column], errors="coerce").fillna(default).max())

    def aggregate_any(row, column: str) -> bool:
        start = int(row["source_index_start"]) if pd.notna(row.get("source_index_start")) else None
        end = int(row["source_index_end"]) if pd.notna(row.get("source_index_end")) else None
        if start is None or end is None or column not in analysis.columns:
            return False
        return bool(analysis.loc[start:end, column].fillna(False).astype(bool).any())

    cleaned["label_iteratif"] = cleaned.apply(aggregate_label, axis=1)
    cleaned["pass_detected"] = cleaned.apply(lambda row: aggregate_max(row, "pass_detected", 0), axis=1)
    cleaned["zone_dense_artefact"] = cleaned.apply(lambda row: aggregate_any(row, "zone_dense_artefact"), axis=1)
    cleaned["suspect_zone_dense"] = cleaned.apply(lambda row: aggregate_any(row, "suspect_zone_dense"), axis=1)
    cleaned["correction_flag_final"] = cleaned["correction_flag"].astype(str)
    cleaned["rr_clean_ms"] = cleaned["rr_interval_ms"].astype(float)
    result.cleaned_frame = cleaned
    return result


def attach_label_2bis(result, label_2bis_values: list[str] | None = None):
    analysis = result.analysis_frame.copy()
    if label_2bis_values is None:
        analysis["label_2bis"] = LABEL_2BIS_NONE
    else:
        analysis["label_2bis"] = pd.Series(label_2bis_values, index=analysis.index, dtype="object")
    result.analysis_frame = analysis

    cleaned = result.cleaned_frame.copy()
    if cleaned.empty:
        result.cleaned_frame = cleaned
        return result

    def aggregate(row):
        start = int(row["source_index_start"]) if pd.notna(row.get("source_index_start")) else None
        end = int(row["source_index_end"]) if pd.notna(row.get("source_index_end")) else None
        if start is None or end is None:
            return LABEL_2BIS_NONE
        values = analysis.loc[start:end, "label_2bis"].fillna(LABEL_2BIS_NONE).astype(str).tolist()
        if LABEL_2BIS_A in values:
            return LABEL_2BIS_A
        if LABEL_2BIS_B in values:
            return LABEL_2BIS_B
        return LABEL_2BIS_NONE

    cleaned["label_2bis"] = cleaned.apply(aggregate, axis=1)
    result.cleaned_frame = cleaned
    return result


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
        values["tolerance_moyen"] = cols[0].number_input("TOLERANCE_MOYEN", 0, 6, defaults.tolerance_moyen, 1)
    with st.expander("Cassures et qualité", expanded=False):
        cols = st.columns(4)
        values["chauffe_gap_court"] = cols[0].number_input("CHAUFFE_GAP_COURT", 1, 12, defaults.chauffe_gap_court, 1)
        values["chauffe_gap_long"] = cols[1].number_input("CHAUFFE_GAP_LONG", 1, 12, defaults.chauffe_gap_long, 1)
        values["seuil_gap_duree_ms"] = cols[1].number_input("SEUIL_GAP_DUREE_MS", 1000, 60000, defaults.seuil_gap_duree_ms, 1000)
        values["seuil_qualite_court"] = cols[2].number_input("SEUIL_QUALITE_COURT", 0.0, 1.0, defaults.seuil_qualite_court, 0.01, format="%.2f")
        values["seuil_qualite_long"] = cols[2].number_input("SEUIL_QUALITE_LONG", 0.0, 1.0, defaults.seuil_qualite_long, 0.01, format="%.2f")
        values["duree_seuil_segment_ms"] = cols[3].number_input("DUREE_SEUIL_SEGMENT_MS", 10000, 600000, defaults.duree_seuil_segment_ms, 1000)
    return RRCleaningParams(**{key: (int(value) if isinstance(getattr(defaults, key), int) else float(value)) for key, value in values.items()})


def build_raw_chart(frame: pd.DataFrame, rr_display_cap_ms: float) -> pd.DataFrame:
    chart = frame.copy()
    chart["t_min"] = chart["t_offset_ms"] / 60000.0
    positive_rr = chart["rr_interval_ms"].where(chart["rr_interval_ms"] > 0, np.nan)
    chart["rr_plot_ms"] = positive_rr.clip(upper=rr_display_cap_ms)

    def series_or_default(column: str, default):
        if column in chart.columns:
            return chart[column]
        return pd.Series(default, index=chart.index)

    chart["label_2bis"] = series_or_default("label_2bis", LABEL_2BIS_NONE)
    chart["label_iteratif"] = series_or_default("label_iteratif", "aucun")
    chart["pass_detected"] = pd.to_numeric(series_or_default("pass_detected", 0), errors="coerce").fillna(0).astype(int)
    chart["zone_dense_artefact"] = series_or_default("zone_dense_artefact", False).fillna(False).astype(bool)
    chart["mrr_local"] = pd.to_numeric(series_or_default("mrr_local", np.nan), errors="coerce")
    chart["qd_drr_local"] = pd.to_numeric(series_or_default("qd_drr_local", np.nan), errors="coerce")
    chart["qd_mrr_local"] = pd.to_numeric(series_or_default("qd_mrr_local", np.nan), errors="coerce")
    chart["th1_local"] = pd.to_numeric(series_or_default("th1_local", np.nan), errors="coerce")
    chart["th2_local"] = pd.to_numeric(series_or_default("th2_local", np.nan), errors="coerce")
    chart["drr_iter"] = pd.to_numeric(series_or_default("drr_iter", np.nan), errors="coerce")
    chart["drr_gap"] = pd.to_numeric(series_or_default("drr_gap", np.nan), errors="coerce")
    chart["rr_clean_ms"] = pd.to_numeric(series_or_default("rr_clean_ms", np.nan), errors="coerce")
    return chart.loc[chart["rr_plot_ms"].notna()].copy()

def build_clean_chart(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chart = frame.copy()
    if chart.empty:
        return chart, chart, chart
    chart["timeline_step_ms"] = chart["timeline_step_ms"].fillna(0.0)
    chart["t_min"] = chart["timeline_step_ms"].cumsum() / 60000.0
    chart["line_group"] = chart["display_rr_ms"].isna().astype(int).cumsum()
    chart["label_2bis"] = chart.get("label_2bis", LABEL_2BIS_NONE)
    line_data = chart.loc[chart["display_rr_ms"].notna()].copy()
    corrected = chart.loc[chart["correction_flag"].isin(["division", "fusion", "interpolation_pchip", "interpolation_lineaire"]) & chart["display_rr_ms"].notna()].copy()
    excluded = chart.loc[chart["display_rr_ms"].isna()].copy()
    return line_data, corrected, excluded


def build_dense_boxes(cleaned_frame: pd.DataFrame, dense_regions_frame: pd.DataFrame) -> pd.DataFrame:
    if cleaned_frame.empty or dense_regions_frame.empty:
        return pd.DataFrame()
    chart = cleaned_frame.copy()
    chart["timeline_step_ms"] = chart["timeline_step_ms"].fillna(0.0)
    chart["t_min"] = chart["timeline_step_ms"].cumsum() / 60000.0
    boxes: list[dict[str, float | int | str]] = []
    fallback_padding = max(float(chart["timeline_step_ms"].replace(0, np.nan).dropna().median()) / 60000.0 if chart["timeline_step_ms"].replace(0, np.nan).notna().any() else 0.02, 0.02)
    for row in dense_regions_frame.itertuples(index=False):
        overlap = chart.loc[
            (chart["source_index_start"].fillna(-1).astype(int) <= int(row.index_fin))
            & (chart["source_index_end"].fillna(-1).astype(int) >= int(row.index_debut))
        ].copy()
        if overlap.empty:
            continue
        visible = overlap.loc[overlap["display_rr_ms"].notna()].copy()
        target = visible if not visible.empty else overlap
        x_start = float(target["t_min"].min())
        x_end = float(target["t_min"].max())
        if x_end <= x_start:
            x_end = x_start + fallback_padding
        y_values = visible["display_rr_ms"].astype(float) if not visible.empty else chart["display_rr_ms"].dropna().astype(float)
        if y_values.empty:
            continue
        y_min = float(y_values.min()) - 45.0
        y_max = float(y_values.max()) + 45.0
        boxes.append({
            "dense_region_id": int(row.dense_region_id),
            "x_start": x_start - fallback_padding,
            "x_end": x_end + fallback_padding,
            "y_min": y_min,
            "y_max": y_max,
            "nb_points": int(row.nb_battements),
            "nb_artefacts": int(row.nb_artefacts),
            "densite_artefact_pct": float(row.densite_artefact_pct),
        })
    return pd.DataFrame(boxes)


def build_dense_zone_summary(cleaned_frame: pd.DataFrame) -> pd.DataFrame:
    if cleaned_frame.empty or "dense_region_id" not in cleaned_frame.columns:
        return pd.DataFrame()
    dense = cleaned_frame.loc[cleaned_frame["dense_region_id"].fillna(0).gt(0)].copy()
    if dense.empty:
        return pd.DataFrame()
    dense["timeline_step_ms"] = dense["timeline_step_ms"].fillna(0.0)
    dense["t_min"] = dense["timeline_step_ms"].cumsum() / 60000.0
    dense["point_nettoye"] = dense["correction_flag"].isin(["division", "fusion", "interpolation_pchip", "interpolation_lineaire"])
    rows: list[dict[str, object]] = []
    for dense_region_id, group in dense.groupby("dense_region_id", sort=True):
        corrections = sorted({value for value in group["correction_flag"].dropna().astype(str) if value not in {"ok", "not_cleaned"}})
        segments = sorted({int(value) for value in group["segment_final_id"].dropna().astype(int)}) if "segment_final_id" in group.columns else []
        start_min = float(group["t_min"].min())
        end_min = float(group["t_min"].max())
        duration_min = max(float(group["timeline_step_ms"].sum()) / 60000.0, end_min - start_min)
        rows.append({
            "dense_region_id": int(dense_region_id),
            "debut_min": round(start_min, 2),
            "fin_min": round(end_min, 2),
            "duree_min": round(duration_min, 2),
            "nb_points": int(len(group)),
            "nb_points_nettoyes": int(group["point_nettoye"].sum()),
            "segments": ", ".join(str(value) for value in segments) if segments else "-",
            "corrections": ", ".join(corrections) if corrections else "aucune",
        })
    return pd.DataFrame(rows)


def counts_frame(items: dict[str, int], label: str) -> pd.DataFrame:
    return pd.DataFrame([{label: key, "nombre": value} for key, value in items.items()])


def build_progress_updater(progress_bar, progress_state: dict[str, float]):
    def update(value: float) -> None:
        bounded = min(max(float(value), 0.0), 1.0)
        if bounded < progress_state["value"]:
            bounded = progress_state["value"]
        progress_state["value"] = bounded
        progress_bar.progress(bounded, text=f"Calcul du nettoyage RR... {bounded * 100:.0f}%")

    return update


def remap_progress(update_progress, start: float, end: float):
    def wrapped(value: float) -> None:
        bounded = min(max(float(value), 0.0), 1.0)
        update_progress(start + ((end - start) * bounded))

    return wrapped


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

    activer_iteratif, iterative_params = build_iterative_inputs()
    params = build_parameter_inputs(RRCleaningParams())
    activer_2bis, params_2bis = build_2bis_inputs()
    session, rr_frame, _ = repository.load_session_data(session_options[selected_label])

    progress_bar = st.progress(0.0, text="Calcul du nettoyage RR... 0%")
    progress_state = {"value": 0.0}
    update_progress = build_progress_updater(progress_bar, progress_state)

    iterative_result = None
    result_standard = None
    result_pre_2bis = None
    try:
        result_standard = attach_label_2bis(
            analyze_rr_artifacts(
                rr_frame,
                params,
                progress_callback=remap_progress(update_progress, 0.00, 0.35 if activer_iteratif else (0.45 if activer_2bis else 1.00)),
            )
        )
        result_pre_2bis = result_standard
        update_progress(0.35 if activer_iteratif else (0.45 if activer_2bis else 1.00))

        if activer_iteratif:
            iterative_result = iterative_reclassification(
                result_standard.analysis_frame,
                max_iter=int(iterative_params["max_iter"]),
                dense_threshold=float(iterative_params["dense_threshold"]),
                dense_window=int(iterative_params["dense_window"]),
                qd_floor=float(iterative_params["qd_floor"]),
                k_drr_max=int(iterative_params["k_drr_max"]),
                seuil_rendement=int(iterative_params["seuil_rendement"]),
                alpha=float(params.alpha),
                params=params,
                progress_callback=remap_progress(update_progress, 0.35, 0.75 if activer_2bis else 1.00),
            )
            result_pre_2bis = attach_label_2bis(iterative_result.result)

        if activer_2bis:
            rr_brut = rr_frame["rr_interval_ms"].astype("float64").to_numpy()
            labels_lipponen = result_pre_2bis.analysis_frame["label"].astype(str).tolist()
            update_progress(0.80)
            labels_modifies, label_2bis_col = run_etape_2bis(rr_brut, labels_lipponen, params=params_2bis)
            update_progress(0.85)
            result = attach_label_2bis(
                analyze_rr_artifacts(
                    rr_frame,
                    params,
                    labels_override=labels_modifies,
                    progress_callback=remap_progress(update_progress, 0.85, 1.00),
                ),
                label_2bis_col,
            )
            if iterative_result is not None:
                result = attach_iterative_overlay(result, iterative_result)
            clean_paths = None
        else:
            result = result_pre_2bis
            clean_paths = None if activer_iteratif else export_clean_result(repository, session, result, params)
        update_progress(1.0)
    except Exception:
        progress_bar.empty()
        raise

    if clean_paths is None:
        if activer_iteratif and activer_2bis:
            st.caption("Modes it?ratif et 2 bis actifs : aucun export clean persistant n'est ecrit.")
        elif activer_iteratif:
            st.caption("Mode it?ratif actif : aucun export clean persistant n'est ecrit.")
        else:
            st.caption("Mode 2 bis actif : aucun export clean persistant n'est ecrit.")
    else:
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

    if activer_2bis:
        n_long_A = int(((result.analysis_frame["label_2bis"] == LABEL_2BIS_A) & (result.analysis_frame["label"] == "long")).sum())
        n_total = int((result.analysis_frame["label_2bis"] != LABEL_2BIS_NONE).sum())
        metrics_2bis = st.columns(2)
        metrics_2bis[0].metric("Artefacts suppl?mentaires (passe A)", str(n_long_A))
        metrics_2bis[1].metric("Beats modifi?s au total", str(n_total))

    if activer_iteratif and iterative_result is not None:
        iterative_metrics = st.columns(4)
        iterative_metrics[0].metric("Artefacts pass 1", str(iterative_result.base_artifacts))
        iterative_metrics[1].metric("Nouveaux artefacts it?ratifs", str(sum(item.new_artifacts for item in iterative_result.pass_summaries)))
        iterative_metrics[2].metric("Zones denses (dernier pass)", str(iterative_result.dense_points_last))
        iterative_metrics[3].metric("Points suspects", str(iterative_result.suspect_points_last))

    counts_col1, counts_col2, counts_col3, counts_col4, counts_col5 = st.columns(5)
    with counts_col1:
        render_section_label("LABELS")
        st.dataframe(counts_frame(result.label_counts, "label"), use_container_width=True, hide_index=True)
    with counts_col2:
        render_section_label("RUN_FLAGS")
        st.dataframe(counts_frame(result.run_flag_counts, "run_flag"), use_container_width=True, hide_index=True)
    with counts_col3:
        render_section_label("RUN_SERIES")
        st.dataframe(counts_frame(result.run_series_flag_counts, "run_series_flag"), use_container_width=True, hide_index=True)
    with counts_col4:
        render_section_label("DECO_FLAGS")
        st.dataframe(counts_frame(result.deco_flag_counts, "deco_flag"), use_container_width=True, hide_index=True)
    with counts_col5:
        render_section_label("CORRECTION_FLAGS")
        st.dataframe(counts_frame(result.correction_counts, "correction_flag"), use_container_width=True, hide_index=True)

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        render_section_label("Segments initiaux")
        st.dataframe(result.segments_frame, use_container_width=True, hide_index=True)
    with info_col2:
        render_section_label("Qualit? des segments")
        st.dataframe(result.quality_segments_frame, use_container_width=True, hide_index=True)
    with info_col3:
        render_section_label("Cassures")
        if result.breaks_frame.empty:
            st.info("Aucune cassure d?tect?e.")
        else:
            st.dataframe(result.breaks_frame, use_container_width=True, hide_index=True)
    with info_col4:
        render_section_label("Zones denses")
        if result.dense_regions_frame.empty:
            st.info("Aucune zone dense d?tect?e.")
        else:
            st.dataframe(result.dense_regions_frame, use_container_width=True, hide_index=True)


    raw_display_cap_ms = max(float(params.rr_max_ms) * 1.25, 1800.0)
    raw_points = build_raw_chart(result.analysis_frame, raw_display_cap_ms)
    clean_line, clean_corrected, clean_excluded = build_clean_chart(result.cleaned_frame)
    dense_boxes = build_dense_boxes(result.cleaned_frame, result.dense_regions_frame)
    base_clean_line, _, _ = build_clean_chart(result_pre_2bis.cleaned_frame) if activer_2bis and result_pre_2bis is not None else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    raw_label_points = raw_points.loc[raw_points["label"].ne(LABEL_OK)].copy()
    raw_2bis_points = raw_points.loc[raw_points["label_2bis"].ne(LABEL_2BIS_NONE)].copy() if activer_2bis else pd.DataFrame()
    raw_iter_points = raw_points.loc[raw_points["pass_detected"].ge(2) & raw_points["label_iteratif"].ne("aucun")].copy() if activer_iteratif else pd.DataFrame()
    if not raw_iter_points.empty:
        raw_iter_points["pass_label"] = raw_iter_points["pass_detected"].map(lambda value: f"Pass {int(value)}")

    raw_line_chart = alt.Chart(raw_points).mark_line(
        color="#9ca3af",
        strokeWidth=1.8,
        opacity=0.95,
    ).encode(
        x=alt.X("t_min:Q", title="Temps (min)"),
        y=alt.Y("rr_plot_ms:Q", title="RR brut (ms)", scale=alt.Scale(domain=[0, raw_display_cap_ms * 1.02])),
        tooltip=["t_min:Q", "rr_interval_ms:Q", "label:N", "label_2bis:N", "run_flag:N", "run_series_flag:N", "deco_flag:N"],
    )
    raw_chart = raw_line_chart
    if not raw_label_points.empty:
        raw_label_scale = alt.Scale(domain=NON_OK_LABEL_ORDER, range=[RAW_LABEL_COLORS[label] for label in NON_OK_LABEL_ORDER])
        raw_label_chart = alt.Chart(raw_label_points).mark_point(
            filled=True,
            size=72,
            stroke="#ffffff",
            strokeWidth=1.0,
            opacity=0.98,
        ).encode(
            x="t_min:Q",
            y="rr_plot_ms:Q",
            color=alt.Color("label:N", scale=raw_label_scale, title="LABEL"),
            tooltip=["t_min:Q", "rr_interval_ms:Q", "label:N", "label_2bis:N", "run_flag:N", "run_series_flag:N", "deco_flag:N"],
        )
        raw_chart = raw_chart + raw_label_chart
    if activer_iteratif and not raw_iter_points.empty:
        raw_iter_chart = alt.Chart(raw_iter_points).mark_point(
            filled=True,
            size=220,
            stroke="#ffffff",
            strokeWidth=1.1,
            opacity=0.96,
        ).encode(
            x="t_min:Q",
            y="rr_plot_ms:Q",
            color=alt.Color("pass_label:N", scale=alt.Scale(domain=ITERATIVE_PASS_DOMAIN, range=ITERATIVE_PASS_COLORS), title="Pass it?ratif"),
            shape=alt.Shape("pass_label:N", scale=alt.Scale(domain=ITERATIVE_PASS_DOMAIN, range=ITERATIVE_PASS_SHAPES), title="Pass it?ratif"),
            tooltip=[
                alt.Tooltip("t_min:Q", title="Temps (min)", format=".3f"),
                alt.Tooltip("rr_interval_ms:Q", title="RR original (ms)", format=".1f"),
                alt.Tooltip("label_iteratif:N", title="label"),
                alt.Tooltip("pass_detected:Q", title="pass_detected"),
                alt.Tooltip("drr_iter:Q", title="dRR_iter", format=".2f"),
                alt.Tooltip("drr_gap:Q", title="dRR_gap", format=".0f"),
                alt.Tooltip("mrr_local:Q", title="mRR_local", format=".2f"),
                alt.Tooltip("th1_local:Q", title="Th1_local", format=".2f"),
                alt.Tooltip("th2_local:Q", title="Th2_local", format=".2f"),
                alt.Tooltip("rr_clean_ms:Q", title="rr_corrig?", format=".2f"),
            ],
        )
        raw_chart = raw_chart + raw_iter_chart
    if activer_2bis and not raw_2bis_points.empty:
        raw_2bis_chart = alt.Chart(raw_2bis_points).mark_point(
            filled=True,
            size=170,
            stroke="#ffffff",
            strokeWidth=1.0,
            opacity=0.95,
        ).encode(
            x="t_min:Q",
            y="rr_plot_ms:Q",
            color=alt.Color("label_2bis:N", scale=RAW_2BIS_COLOR_SCALE, legend=None),
            shape=alt.Shape("label_2bis:N", scale=RAW_2BIS_SHAPE_SCALE, title="LABEL 2 bis"),
            tooltip=["t_min:Q", "rr_interval_ms:Q", "label:N", "label_2bis:N", "run_flag:N", "run_series_flag:N", "deco_flag:N"],
        )
        raw_chart = raw_chart + raw_2bis_chart
    if activer_2bis and not base_clean_line.empty:
        clean_base_line_chart = alt.Chart(base_clean_line).mark_line(color="#94a3b8", strokeWidth=1.6, strokeDash=[8, 4]).encode(
            x=alt.X("t_min:Q", title="Temps cumule clean (min)"),
            y=alt.Y("display_rr_ms:Q", title="RR clean (ms)"),
            detail="line_group:N",
            order=alt.Order("t_min:Q"),
        )
    else:
        clean_base_line_chart = None
    clean_line_chart = alt.Chart(clean_line).mark_line(color="#2d5f43", strokeWidth=2.2).encode(
        x=alt.X("t_min:Q", title="Temps cumul? clean (min)"),
        y=alt.Y("display_rr_ms:Q", title="RR clean (ms)"),
        detail="line_group:N",
        order=alt.Order("t_min:Q"),
    )
    present_corrections = [
        method
        for method in ["division", "fusion", "interpolation_pchip", "interpolation_lineaire"]
        if method in clean_corrected["correction_flag"].astype(str).unique().tolist()
    ]
    correction_colors = {
        "division": "#1d4ed8",
        "fusion": "#b45309",
        "interpolation_pchip": "#be123c",
        "interpolation_lineaire": "#7c3aed",
    }
    correction_scale = alt.Scale(
        domain=present_corrections,
        range=[correction_colors[method] for method in present_corrections],
    ) if present_corrections else alt.Scale(domain=["division"], range=["#1d4ed8"])
    clean_corrected_chart = alt.Chart(clean_corrected).mark_point(filled=True, size=90, stroke="#ffffff", strokeWidth=1.2).encode(
        x="t_min:Q",
        y="display_rr_ms:Q",
        color=alt.Color("correction_flag:N", scale=correction_scale, title="Correction"),
        shape=alt.Shape(
            "correction_flag:N",
            scale=alt.Scale(
                domain=["division", "fusion", "interpolation_pchip", "interpolation_lineaire"],
                range=["circle", "square", "diamond", "triangle-up"],
            ),
            title="Correction",
        ),
        tooltip=["t_min:Q", "rr_interval_ms:Q", "label:N", "label_2bis:N", "run_flag:N", "run_series_flag:N", "correction_flag:N", "fc_ok:N", "hrr_ok:N", "rmssd_ok:N"],
    )
    clean_excluded_chart = alt.Chart(clean_excluded).mark_tick(thickness=2, size=18, color="#475569").encode(
        x="t_min:Q",
        tooltip=["t_min:Q", "label:N", "label_2bis:N", "run_flag:N", "run_series_flag:N", "correction_flag:N", "segment_status:N"],
    )
    dense_outline = alt.Chart(dense_boxes).mark_rect(stroke="#0f766e", strokeWidth=3.0, strokeDash=[8, 4], color="#14b8a6", fillOpacity=0.10).encode(
        x="x_start:Q", x2="x_end:Q", y="y_min:Q", y2="y_max:Q", tooltip=["dense_region_id:Q", "nb_points:Q", "nb_artefacts:Q", "densite_artefact_pct:Q"]
    )

    clean_chart = dense_outline + clean_line_chart + clean_corrected_chart + clean_excluded_chart
    if clean_base_line_chart is not None:
        clean_chart = dense_outline + clean_base_line_chart + clean_line_chart + clean_corrected_chart + clean_excluded_chart

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        render_section_label("RR bruts")
        st.altair_chart(raw_chart.properties(height=320).interactive(), use_container_width=True)
        if activer_iteratif and activer_2bis:
            st.caption("La ligne relie tous les RR bruts. Les labels non ok restent mis en avant, les passes it?ratives apparaissent avec un marqueur d?di? par pass, et les marqueurs 2 bis ne s'affichent que pour label_2bis non aucun.")
        elif activer_iteratif:
            st.caption("La ligne relie tous les RR bruts. Les labels non ok restent mis en avant et les nouveaux artefacts it?ratifs sont superpos?s avec un marqueur d?di? par pass.")
        elif activer_2bis:
            st.caption("La ligne relie tous les RR bruts. Seuls les points avec label non ok sont mis en avant, et seuls les points avec label_2bis non aucun recoivent un marqueur 2 bis.")
        else:
            st.caption("La ligne relie tous les RR bruts. Seuls les points avec label non ok sont mis en avant.")
    with chart_col2:
        render_section_label("RR clean")
        st.altair_chart(clean_chart.properties(height=320).interactive(), use_container_width=True)
        st.caption("Les zones denses run_serie sont encadr?es en vert d'eau. Les points nettoy?s visibles utilisent une couleur et une forme distinctes selon la m?thode de correction. Quand l'?tape 2 bis est active, la courbe grise en tirets montre le RR clean pr?-2 bis et la courbe verte le RR clean post-2 bis.")

    if activer_iteratif and iterative_result is not None:
        pass_lines = "\n".join(
            f"- Pass {item.pass_number} : {item.new_artifacts} nouveaux artefacts | zones denses : {item.dense_points} points ({item.dense_pct:.1f}%) | suspects : {item.suspect_points}" + (f" | arr?t : {item.stop_reason}" if item.stop_reason else "")
            for item in iterative_result.pass_summaries
        )
        st.markdown(
            f"""
**R?sultat it?ration** :
- Pass 1 : {iterative_result.base_artifacts} artefacts d?tect?s
{pass_lines}
- Convergence atteinte au pass {iterative_result.final_pass} ({iterative_result.stop_reason})
- Zones denses (derni?re ?valuation) : {iterative_result.dense_points_last} battements ({iterative_result.dense_pct_last:.1f}%)
- Points suspects non classifi?s : {iterative_result.suspect_points_last}
"""
        )

    render_section_label("Variables interm?diaires")
    variable_columns = [
        "t_offset_ms", "rr_interval_ms", "source_pipeline_flag", "deco_flag", "label_initial", "label", "label_2bis", "label_iteratif", "pass_detected", "run_flag", "run_series_flag",
        "dRR_ms", "Th1_ms", "dRR_norm", "med_locale_ms", "mRR_brut_ms", "mRR_ms", "Th2_ms", "mRR_norm", "S21", "S22",
        "mrr_local", "qd_drr_local", "qd_mrr_local", "th1_local", "th2_local", "drr_iter", "drr_gap", "zone_dense_artefact", "suspect_zone_dense", "correction_flag_final", "rr_clean_ms",
        "initial_segment_id", "initial_segment_status", "segment_final_id", "dense_region_id", "correction_flag_raw",
    ]
    available = [column for column in variable_columns if column in result.analysis_frame.columns]
    st.dataframe(result.analysis_frame[available], use_container_width=True, height=320)

    table_col1, table_col2 = st.columns(2)
    with table_col1:
        render_section_label("Table RR bruts")
        raw_columns = [
            "t_offset_ms", "rr_interval_ms", "source_pipeline_flag", "deco_flag", "label", "label_2bis", "label_iteratif", "pass_detected",
            "run_flag", "run_series_flag", "initial_segment_id", "initial_segment_status", "segment_final_id", "dense_region_id",
            "mrr_local", "qd_drr_local", "qd_mrr_local", "th1_local", "th2_local", "drr_iter", "drr_gap",
            "zone_dense_artefact", "suspect_zone_dense", "correction_flag_raw", "correction_flag_final", "rr_clean_ms",
        ]
        raw_available = [column for column in raw_columns if column in result.analysis_frame.columns]
        raw_table = result.analysis_frame[raw_available]
        st.dataframe(raw_table, use_container_width=True, height=340)
    with table_col2:
        render_section_label("Table RR clean")
        clean_table_source = result.cleaned_frame.assign(
            point_nettoye=result.cleaned_frame["correction_flag"].isin(["division", "fusion", "interpolation_pchip", "interpolation_lineaire"]),
            zone_dense_artefact=result.cleaned_frame.get("zone_dense_artefact", result.cleaned_frame["dense_region_id"].fillna(0).gt(0)),
            suspect_zone_dense=result.cleaned_frame.get("suspect_zone_dense", False),
            correction_flag_final=result.cleaned_frame.get("correction_flag_final", result.cleaned_frame["correction_flag"].astype(str)),
            rr_clean_ms=pd.to_numeric(result.cleaned_frame.get("rr_clean_ms", result.cleaned_frame["rr_interval_ms"]), errors="coerce"),
        )
        clean_columns = [
            "cleaned_index", "rr_interval_ms", "rr_clean_ms", "label", "label_2bis", "label_iteratif", "pass_detected",
            "run_flag", "run_series_flag", "deco_flag", "correction_flag", "correction_flag_final", "point_nettoye",
            "zone_dense_artefact", "suspect_zone_dense", "fc_ok", "hrr_ok", "rmssd_ok", "segment_status",
            "quality_segment_label", "dense_region_id", "source_reference",
        ]
        clean_available = [column for column in clean_columns if column in clean_table_source.columns]
        cleaned_table = clean_table_source[clean_available] if show_full_tables else clean_table_source[clean_available].head(700)
        st.dataframe(cleaned_table, use_container_width=True, height=340)



if __name__ == "__main__":
    main()
