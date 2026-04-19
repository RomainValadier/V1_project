from __future__ import annotations

import json
import os
from dataclasses import asdict

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
from polar_app.iterative_lipponen import IterativeLipponenResult, IterativePassSummary, PASS_MARKERS, iterative_reclassification
from polar_app.repository import ProcessedSessionRepository
from polar_app.rr_clean_component import rr_clean_viewer
from polar_app.rr_manual_component import rr_manual_editor
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
    RRCleaningResult,
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
MANUAL_FLAG_OK = "ok"
MANUAL_FLAG_VALUE = "manuel"
MANUAL_LABEL_COLOR = "#ec4899"
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


def rr_manual_event_key(session_id: str) -> str:
    return f"rr_manual_event_{session_id}"


def _clone_rr_cleaning_result(result: RRCleaningResult) -> RRCleaningResult:
    return RRCleaningResult(
        analysis_frame=result.analysis_frame.copy(deep=True),
        cleaned_frame=result.cleaned_frame.copy(deep=True),
        segments_frame=result.segments_frame.copy(deep=True),
        quality_segments_frame=result.quality_segments_frame.copy(deep=True),
        breaks_frame=result.breaks_frame.copy(deep=True),
        dense_regions_frame=result.dense_regions_frame.copy(deep=True),
        label_counts=dict(result.label_counts),
        run_flag_counts=dict(result.run_flag_counts),
        run_series_flag_counts=dict(result.run_series_flag_counts),
        deco_flag_counts=dict(result.deco_flag_counts),
        correction_counts=dict(result.correction_counts),
        total_points=int(result.total_points),
        cleaned_points=int(result.cleaned_points),
        ok_rr_total=int(result.ok_rr_total),
        non_viable_rr_total=int(result.non_viable_rr_total),
        n_cassures_total=int(result.n_cassures_total),
        n_cassures_deco=int(result.n_cassures_deco),
        n_cassures_artefact=int(result.n_cassures_artefact),
        n_zones_denses=int(result.n_zones_denses),
        dense_points_count=int(result.dense_points_count),
        n_segments_actifs=int(result.n_segments_actifs),
        n_segments_exclus=int(result.n_segments_exclus),
        correction_rate_outside_breaks=float(result.correction_rate_outside_breaks),
        global_non_ok_rate=float(result.global_non_ok_rate),
        global_quality_label=str(result.global_quality_label),
    )


def _clone_iterative_result(iterative_result: IterativeLipponenResult | None) -> IterativeLipponenResult | None:
    if iterative_result is None:
        return None
    return IterativeLipponenResult(
        result=_clone_rr_cleaning_result(iterative_result.result),
        final_labels=list(iterative_result.final_labels),
        pass_summaries=[
            IterativePassSummary(
                pass_number=int(item.pass_number),
                new_artifacts=int(item.new_artifacts),
                dense_points=int(item.dense_points),
                dense_pct=float(item.dense_pct),
                suspect_points=int(item.suspect_points),
                stop_reason=str(item.stop_reason),
            )
            for item in iterative_result.pass_summaries
        ],
        final_pass=int(iterative_result.final_pass),
        stop_reason=str(iterative_result.stop_reason),
        base_artifacts=int(iterative_result.base_artifacts),
        dense_points_last=int(iterative_result.dense_points_last),
        dense_pct_last=float(iterative_result.dense_pct_last),
        suspect_points_last=int(iterative_result.suspect_points_last),
    )


@st.cache_data(show_spinner=False)
def _compute_auto_pipeline_cached(session_id: str, rr_frame: pd.DataFrame, params_payload: str, activer_iteratif: bool, iterative_params_payload: str, activer_2bis: bool, params_2bis_payload: str):
    del session_id
    params = RRCleaningParams(**json.loads(params_payload))
    iterative_params = json.loads(iterative_params_payload)
    params_2bis = json.loads(params_2bis_payload)

    result_standard = attach_label_2bis(analyze_rr_artifacts(rr_frame, params))
    result_pre_2bis = result_standard
    iterative_result = None
    label_2bis_values = None

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
            progress_callback=None,
        )
        result_pre_2bis = attach_label_2bis(iterative_result.result)

    if activer_2bis:
        rr_brut = rr_frame["rr_interval_ms"].astype("float64").to_numpy()
        labels_lipponen = result_pre_2bis.analysis_frame["label"].astype(str).tolist()
        labels_modifies, label_2bis_values = run_etape_2bis(rr_brut, labels_lipponen, params=params_2bis)
        result_auto = attach_label_2bis(
            analyze_rr_artifacts(
                rr_frame,
                params,
                labels_override=labels_modifies,
                progress_callback=None,
            ),
            label_2bis_values,
        )
    else:
        result_auto = result_pre_2bis

    if iterative_result is not None and result_auto is not iterative_result.result:
        result_auto = attach_iterative_overlay(result_auto, iterative_result)

    label_auto_values = result_auto.analysis_frame["label"].astype(str).tolist()
    return {
        "result_standard": _clone_rr_cleaning_result(result_standard),
        "result_pre_2bis": _clone_rr_cleaning_result(result_pre_2bis),
        "result_auto": _clone_rr_cleaning_result(result_auto),
        "iterative_result": _clone_iterative_result(iterative_result),
        "label_2bis_values": list(label_2bis_values) if label_2bis_values is not None else None,
        "label_auto_values": list(label_auto_values),
    }


@st.cache_data(show_spinner=False)
def _compute_manual_override_pipeline_cached(session_id: str, rr_frame: pd.DataFrame, params_payload: str, manual_override_payload: str) -> RRCleaningResult:
    del session_id
    params = RRCleaningParams(**json.loads(params_payload))
    manual_override_labels = list(json.loads(manual_override_payload))
    result = analyze_rr_artifacts(
        rr_frame,
        params,
        labels_override=manual_override_labels,
        progress_callback=None,
    )
    return _clone_rr_cleaning_result(result)


def _sanitize_component_value(value):
    if isinstance(value, dict):
        return {key: _sanitize_component_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_sanitize_component_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_component_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _aggregate_string(values: list[str], default: str) -> str:
    filtered = [value for value in values if value and value != default]
    if not filtered:
        return default
    unique = list(dict.fromkeys(filtered))
    return unique[0] if len(unique) == 1 else "mixte"


def attach_manual_overlay(result, label_auto_values: list[str], manual_annotations: list[dict[str, object]]):
    analysis = result.analysis_frame.copy()
    manual_indices = {int(item["source_index"]) for item in manual_annotations if "source_index" in item}
    analysis["label_auto"] = pd.Series(label_auto_values, index=analysis.index, dtype="object")
    analysis["manual_flag"] = MANUAL_FLAG_OK
    analysis["manual_treated_as"] = ""
    analysis["is_manual_annotation"] = False
    for idx in sorted(manual_indices):
        if 0 <= idx < len(analysis):
            analysis.at[idx, "manual_flag"] = MANUAL_FLAG_VALUE
            analysis.at[idx, "manual_treated_as"] = LABEL_LONG
            analysis.at[idx, "is_manual_annotation"] = True
            analysis.at[idx, "label"] = MANUAL_FLAG_VALUE
    analysis["correction_flag_final"] = _map_clean_text_from_cleaned(result.cleaned_frame, len(analysis), "correction_flag", "ok").astype(str)
    analysis["rr_clean_ms"] = _map_rr_clean_from_cleaned(result.cleaned_frame, len(analysis))
    result.analysis_frame = analysis

    cleaned = result.cleaned_frame.copy()
    if cleaned.empty:
        result.cleaned_frame = cleaned
        return result

    def aggregate_manual_flag(row):
        start = int(row["source_index_start"]) if pd.notna(row.get("source_index_start")) else None
        end = int(row["source_index_end"]) if pd.notna(row.get("source_index_end")) else None
        if start is None or end is None:
            return MANUAL_FLAG_OK
        values = analysis.loc[start:end, "manual_flag"].fillna(MANUAL_FLAG_OK).astype(str).tolist()
        return MANUAL_FLAG_VALUE if MANUAL_FLAG_VALUE in values else MANUAL_FLAG_OK

    def aggregate_manual_bool(row):
        start = int(row["source_index_start"]) if pd.notna(row.get("source_index_start")) else None
        end = int(row["source_index_end"]) if pd.notna(row.get("source_index_end")) else None
        if start is None or end is None:
            return False
        return bool(analysis.loc[start:end, "is_manual_annotation"].fillna(False).astype(bool).any())

    cleaned["manual_flag"] = cleaned.apply(aggregate_manual_flag, axis=1)
    cleaned["is_manual_annotation"] = cleaned.apply(aggregate_manual_bool, axis=1)
    cleaned["manual_treated_as"] = np.where(cleaned["manual_flag"].eq(MANUAL_FLAG_VALUE), LABEL_LONG, "")
    cleaned["correction_flag_final"] = cleaned["correction_flag"].astype(str)
    cleaned["rr_clean_ms"] = cleaned["rr_interval_ms"].astype(float)
    result.cleaned_frame = cleaned
    return result


def build_manual_override_labels(base_labels: list[str], manual_annotations: list[dict[str, object]]) -> list[str]:
    labels = list(base_labels)
    for item in manual_annotations:
        if "source_index" not in item:
            continue
        idx = int(item["source_index"])
        if 0 <= idx < len(labels):
            labels[idx] = LABEL_LONG
    return labels


def build_rr_manual_component_points(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    points = frame.copy().reset_index().rename(columns={"index": "source_index"})
    columns = [
        "source_index", "t_min", "rr_plot_ms", "rr_interval_ms", "label", "label_auto", "label_2bis",
        "label_iteratif", "pass_detected", "manual_flag",
    ]
    available = [column for column in columns if column in points.columns]
    payload = points[available].copy()
    for column in ["label", "label_auto", "label_2bis", "label_iteratif", "manual_flag"]:
        if column not in payload.columns:
            payload[column] = ""
    if "pass_detected" not in payload.columns:
        payload["pass_detected"] = 0
    return _sanitize_component_value(payload.to_dict(orient="records"))


def build_manual_analysis_table(analysis_frame: pd.DataFrame) -> pd.DataFrame:
    if analysis_frame.empty or "manual_flag" not in analysis_frame.columns:
        return pd.DataFrame()
    manual = analysis_frame.loc[analysis_frame["manual_flag"].eq(MANUAL_FLAG_VALUE)].copy()
    if manual.empty:
        return manual
    manual = manual.reset_index().rename(columns={"index": "source_index"})
    columns = [
        "source_index", "t_offset_ms", "rr_interval_ms", "label_auto", "label", "manual_treated_as",
        "deco_flag", "run_flag", "run_series_flag", "dRR_ms", "Th1_ms", "dRR_norm", "med_locale_ms",
        "mRR_brut_ms", "mRR_ms", "Th2_ms", "mRR_norm", "S21", "S22", "segment_final_id", "dense_region_id",
        "correction_flag_final", "rr_clean_ms",
    ]
    available = [column for column in columns if column in manual.columns]
    return manual[available].copy()


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
    chart["label_auto"] = series_or_default("label_auto", chart.get("label", LABEL_OK))
    chart["manual_flag"] = series_or_default("manual_flag", MANUAL_FLAG_OK)
    chart["manual_treated_as"] = series_or_default("manual_treated_as", "")
    chart["is_manual_annotation"] = series_or_default("is_manual_annotation", False).fillna(False).astype(bool)
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


def restore_excluded_clean_display(excluded_frame: pd.DataFrame, analysis_frame: pd.DataFrame, reference_line_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    if excluded_frame.empty:
        return excluded_frame
    restored = excluded_frame.copy()
    restored["display_rr_ms"] = pd.to_numeric(restored.get("display_rr_ms", np.nan), errors="coerce")
    if analysis_frame.empty or "rr_interval_ms" not in analysis_frame.columns:
        return restored
    raw_rr = pd.to_numeric(analysis_frame["rr_interval_ms"], errors="coerce")
    reference_values = pd.Series(dtype="float64")
    if reference_line_frame is not None and not reference_line_frame.empty and "display_rr_ms" in reference_line_frame.columns:
        reference_values = pd.to_numeric(reference_line_frame["display_rr_ms"], errors="coerce")
    reference_values = reference_values.where(reference_values > 0, np.nan).dropna()
    if reference_values.empty:
        reference_values = raw_rr.where((raw_rr > 0) & (raw_rr < 4000), np.nan).dropna()
    if reference_values.empty:
        display_cap = 1800.0
        display_default = 900.0
    else:
        display_cap = max(float(np.nanpercentile(reference_values, 99)) * 1.25, float(np.nanmedian(reference_values)) * 1.8, 900.0)
        display_default = float(np.nanmedian(reference_values))

    def resolve_display(row):
        current = pd.to_numeric(pd.Series([row.get("display_rr_ms")]), errors="coerce").iloc[0]
        if pd.notna(current):
            return float(min(current, display_cap))
        start = int(row["source_index_start"]) if pd.notna(row.get("source_index_start")) else None
        end = int(row["source_index_end"]) if pd.notna(row.get("source_index_end")) else None
        if start is None or end is None or start < 0:
            return display_default
        bounded_end = min(end, len(raw_rr) - 1)
        segment = raw_rr.iloc[start : bounded_end + 1].where(raw_rr.iloc[start : bounded_end + 1] > 0, np.nan).dropna()
        segment_under_cap = segment.loc[segment.le(display_cap)]
        if not segment_under_cap.empty:
            return float(segment_under_cap.iloc[-1])
        window_start = max(0, start - 3)
        window_end = min(len(raw_rr) - 1, bounded_end + 3)
        neighborhood = raw_rr.iloc[window_start : window_end + 1].where((raw_rr.iloc[window_start : window_end + 1] > 0) & (raw_rr.iloc[window_start : window_end + 1] <= display_cap), np.nan).dropna()
        if not neighborhood.empty:
            return float(neighborhood.iloc[-1])
        if not segment.empty:
            return float(min(segment.iloc[-1], display_cap))
        return display_default

    restored["display_rr_ms"] = restored.apply(resolve_display, axis=1)
    return restored


def merge_touching_excluded_zones(excluded_frame: pd.DataFrame) -> pd.DataFrame:
    if excluded_frame.empty:
        return excluded_frame
    frame = excluded_frame.copy()
    if "source_index_start" not in frame.columns or "source_index_end" not in frame.columns:
        return frame
    frame["source_index_start"] = pd.to_numeric(frame["source_index_start"], errors="coerce")
    frame["source_index_end"] = pd.to_numeric(frame["source_index_end"], errors="coerce")
    frame = frame.loc[frame["source_index_start"].notna() & frame["source_index_end"].notna()].copy()
    if frame.empty:
        return excluded_frame
    frame = frame.sort_values(["source_index_start", "source_index_end", "t_min"], kind="stable").reset_index(drop=True)

    merged_rows: list[dict[str, object]] = []
    current_rows: list[pd.Series] = []
    current_end: int | None = None

    def finalize_group(rows: list[pd.Series]) -> None:
        if not rows:
            return
        group = pd.DataFrame(rows)
        start = int(group["source_index_start"].min())
        end = int(group["source_index_end"].max())
        t_values = pd.to_numeric(group.get("t_min", np.nan), errors="coerce").dropna()
        y_values = pd.to_numeric(group.get("display_rr_ms", np.nan), errors="coerce").dropna()
        rr_values = pd.to_numeric(group.get("rr_interval_ms", np.nan), errors="coerce").dropna()

        def aggregate_text(column: str, default: str = "") -> str:
            if column not in group.columns:
                return default
            values = [str(value) for value in group[column].dropna().astype(str).tolist() if str(value)]
            if not values:
                return default
            unique = list(dict.fromkeys(values))
            return unique[0] if len(unique) == 1 else "mixte"

        merged_rows.append({
            "t_min": float((t_values.min() + t_values.max()) / 2.0) if not t_values.empty else np.nan,
            "x_start_min": float(t_values.min()) if not t_values.empty else np.nan,
            "x_end_min": float(t_values.max()) if not t_values.empty else np.nan,
            "display_rr_ms": float(y_values.median()) if not y_values.empty else np.nan,
            "rr_interval_ms": float(rr_values.median()) if not rr_values.empty else np.nan,
            "label": aggregate_text("label"),
            "label_2bis": aggregate_text("label_2bis"),
            "line_group": int(pd.to_numeric(group.get("line_group", 0), errors="coerce").dropna().iloc[0]) if "line_group" in group.columns and not pd.to_numeric(group.get("line_group", 0), errors="coerce").dropna().empty else 0,
            "run_flag": aggregate_text("run_flag"),
            "run_series_flag": aggregate_text("run_series_flag"),
            "correction_flag": aggregate_text("correction_flag", "not_cleaned"),
            "segment_status": aggregate_text("segment_status"),
            "source_index_start": start,
            "source_index_end": end,
            "source_reference": f"{start + 1}-{end + 1}" if end > start else f"{start + 1}",
        })

    for row in frame.itertuples(index=False):
        start = int(row.source_index_start)
        end = int(row.source_index_end)
        if current_end is None or start > (current_end + 1):
            finalize_group(current_rows)
            current_rows = [pd.Series(row._asdict())]
            current_end = end
            continue
        current_rows.append(pd.Series(row._asdict()))
        current_end = max(int(current_end), end)

    finalize_group(current_rows)
    return pd.DataFrame(merged_rows)


def build_excluded_boxes(excluded_frame: pd.DataFrame) -> pd.DataFrame:
    if excluded_frame.empty:
        return pd.DataFrame()
    chart = excluded_frame.copy()
    if "x_start_min" not in chart.columns or "x_end_min" not in chart.columns:
        return pd.DataFrame()
    x_start_values = pd.to_numeric(chart["x_start_min"], errors="coerce").dropna()
    x_end_values = pd.to_numeric(chart["x_end_min"], errors="coerce").dropna()
    if x_start_values.empty or x_end_values.empty:
        return pd.DataFrame()
    boxes: list[dict[str, object]] = []
    for row in chart.itertuples(index=False):
        x_start = pd.to_numeric(pd.Series([getattr(row, "x_start_min", np.nan)]), errors="coerce").iloc[0]
        x_end = pd.to_numeric(pd.Series([getattr(row, "x_end_min", np.nan)]), errors="coerce").iloc[0]
        if pd.isna(x_start) or pd.isna(x_end):
            continue
        if float(x_end) <= float(x_start):
            x_end = float(x_start) + 0.001
        boxes.append({
            "x_start": float(x_start),
            "x_end": float(x_end),
            "label": str(getattr(row, "label", "") or ""),
            "segment_status": str(getattr(row, "segment_status", "") or ""),
            "source_reference": str(getattr(row, "source_reference", "") or ""),
        })
    return pd.DataFrame(boxes)


def _clean_component_records(frame: pd.DataFrame, fallback_display_from_rr: bool = False) -> list[dict[str, object]]:
    if frame.empty:
        return []
    payload = frame.copy()
    if "display_rr_ms" not in payload.columns:
        payload["display_rr_ms"] = np.nan
    if fallback_display_from_rr:
        payload["display_rr_ms"] = pd.to_numeric(payload["display_rr_ms"], errors="coerce")
        rr_fallback = pd.to_numeric(payload["rr_interval_ms"], errors="coerce") if "rr_interval_ms" in payload.columns else pd.Series(np.nan, index=payload.index)
        payload["display_rr_ms"] = payload["display_rr_ms"].where(payload["display_rr_ms"].notna(), rr_fallback)
    columns = [
        "t_min", "display_rr_ms", "rr_interval_ms", "label", "label_2bis",
        "line_group",
        "run_flag", "run_series_flag", "correction_flag", "segment_status",
    ]
    available = [column for column in columns if column in payload.columns]
    data = payload[available].copy()
    for column, default in [("label", ""), ("label_2bis", ""), ("run_flag", ""), ("run_series_flag", ""), ("correction_flag", ""), ("segment_status", "")]:
        if column not in data.columns:
            data[column] = default
    return _sanitize_component_value(data.to_dict(orient="records"))


def build_clean_component_payload(line_frame: pd.DataFrame, corrected_frame: pd.DataFrame, excluded_frame: pd.DataFrame, dense_boxes: pd.DataFrame, excluded_boxes: pd.DataFrame | None = None, base_line_frame: pd.DataFrame | None = None) -> dict[str, object]:
    return {
        "line_points": _clean_component_records(line_frame),
        "corrected_points": _clean_component_records(corrected_frame),
        "excluded_points": _clean_component_records(excluded_frame, fallback_display_from_rr=True),
        "dense_boxes": _sanitize_component_value(dense_boxes.to_dict(orient="records")) if not dense_boxes.empty else [],
        "excluded_boxes": _sanitize_component_value(excluded_boxes.to_dict(orient="records")) if excluded_boxes is not None and not excluded_boxes.empty else [],
        "base_line_points": _clean_component_records(base_line_frame if base_line_frame is not None else pd.DataFrame()),
    }


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
    manual_annotations = repository.get_rr_manual_annotations(session.session_id)

    progress_bar = st.progress(0.0, text="Calcul du nettoyage RR... 0%")
    progress_state = {"value": 0.0}
    update_progress = build_progress_updater(progress_bar, progress_state)

    iterative_result = None
    result_standard = None
    result_pre_2bis = None
    label_2bis_values = None
    params_payload = json.dumps(asdict(params), sort_keys=True, ensure_ascii=True)
    iterative_params_payload = json.dumps(iterative_params, sort_keys=True, ensure_ascii=True)
    params_2bis_payload = json.dumps(params_2bis, sort_keys=True, ensure_ascii=True)
    try:
        update_progress(0.05)
        cached_auto = _compute_auto_pipeline_cached(
            session.session_id,
            rr_frame,
            params_payload,
            activer_iteratif,
            iterative_params_payload,
            activer_2bis,
            params_2bis_payload,
        )
        result_standard = _clone_rr_cleaning_result(cached_auto["result_standard"])
        result_pre_2bis = _clone_rr_cleaning_result(cached_auto["result_pre_2bis"])
        result_auto = _clone_rr_cleaning_result(cached_auto["result_auto"])
        iterative_result = _clone_iterative_result(cached_auto.get("iterative_result"))
        label_2bis_values = list(cached_auto["label_2bis_values"]) if cached_auto.get("label_2bis_values") is not None else None
        label_auto_values = list(cached_auto["label_auto_values"])
        update_progress(0.92)

        manual_override_labels = build_manual_override_labels(label_auto_values, manual_annotations)
        if any(label_left != label_right for label_left, label_right in zip(label_auto_values, manual_override_labels)):
            manual_override_payload = json.dumps(manual_override_labels, ensure_ascii=True)
            result = _compute_manual_override_pipeline_cached(
                session.session_id,
                rr_frame,
                params_payload,
                manual_override_payload,
            )
            result = attach_label_2bis(result, label_2bis_values)
            if iterative_result is not None:
                result = attach_iterative_overlay(result, iterative_result)
            update_progress(1.0)
        else:
            result = _clone_rr_cleaning_result(result_auto)
        result = attach_manual_overlay(result, label_auto_values, manual_annotations)
        update_progress(1.0)
    except Exception:
        progress_bar.empty()
        raise

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
        label_order = LABEL_ORDER + [MANUAL_FLAG_VALUE]
        label_values = result.analysis_frame.get("label", pd.Series(dtype="object")).fillna("").astype(str).value_counts().to_dict()
        label_counts = {key: int(label_values.get(key, 0)) for key in label_order if int(label_values.get(key, 0)) > 0}
        st.dataframe(counts_frame(label_counts, "label"), use_container_width=True, hide_index=True)
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
    raw_component_points = build_rr_manual_component_points(raw_points)
    manual_analysis_table = build_manual_analysis_table(result.analysis_frame)
    clean_line, clean_corrected, clean_excluded = build_clean_chart(result.cleaned_frame)
    clean_excluded = restore_excluded_clean_display(clean_excluded, result.analysis_frame, clean_line)
    clean_excluded = merge_touching_excluded_zones(clean_excluded)
    dense_boxes = build_dense_boxes(result.cleaned_frame, result.dense_regions_frame)
    excluded_boxes = build_excluded_boxes(clean_excluded)
    base_clean_line, _, _ = build_clean_chart(result_pre_2bis.cleaned_frame) if activer_2bis and result_pre_2bis is not None else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    clean_component_payload = build_clean_component_payload(
        clean_line,
        clean_corrected,
        clean_excluded,
        dense_boxes,
        excluded_boxes,
        base_clean_line if activer_2bis and not base_clean_line.empty else pd.DataFrame(),
    )

    render_section_label("RR bruts")
    component_value = rr_manual_editor(
        points=raw_component_points,
        rr_max=raw_display_cap_ms,
        chart_title="RR bruts",
        default=None,
        key=f"rr_manual_editor_{session.session_id}",
    )
    if isinstance(component_value, dict):
        event_id = int(component_value.get("event_id", 0) or 0)
        last_event_id = int(st.session_state.get(rr_manual_event_key(session.session_id), 0) or 0)
        if event_id > last_event_id:
            st.session_state[rr_manual_event_key(session.session_id)] = event_id
            event_type = str(component_value.get("event_type", "") or "")
            if event_type == "apply_pending_selection":
                source_indices = [int(value) for value in (component_value.get("source_indices") or [])]
                repository.apply_rr_manual_annotation_batch(session.session_id, source_indices)
                st.rerun()
            if event_type == "clear_all_manual_annotations":
                repository.clear_rr_manual_annotations(session.session_id)
                st.rerun()
    action_cols = st.columns([1, 1.2])
    action_cols[0].metric("Flags manuels", str(len(manual_annotations)))
    if action_cols[1].button("Effacer les flags manuels", use_container_width=True, disabled=not manual_annotations, key=f"clear_rr_manual_{session.session_id}"):
        repository.clear_rr_manual_annotations(session.session_id)
        st.rerun()
    if activer_iteratif and activer_2bis:
        st.caption("La ligne relie tous les RR bruts. Les labels non ok restent mis en avant, les passes it?ratives restent visibles, les marqueurs 2 bis sont conserv?s, et tu peux pr?parer plusieurs points avant de valider le lot manuel.")
    elif activer_iteratif:
        st.caption("La ligne relie tous les RR bruts. Les labels non ok restent mis en avant, les passes it?ratives sont conserv?es, et tu peux pr?parer plusieurs points avant de valider le lot manuel.")
    elif activer_2bis:
        st.caption("La ligne relie tous les RR bruts. Les labels non ok restent mis en avant, les marqueurs 2 bis sont conserv?s, et tu peux pr?parer plusieurs points avant de valider le lot manuel.")
    else:
        st.caption("La ligne relie tous les RR bruts. Les labels non ok restent mis en avant, et tu peux pr?parer plusieurs points avant de valider le lot manuel.")

    render_section_label("RR clean")
    rr_clean_viewer(
        line_points=clean_component_payload["line_points"],
        corrected_points=clean_component_payload["corrected_points"],
        excluded_points=clean_component_payload["excluded_points"],
        dense_boxes=clean_component_payload["dense_boxes"],
        excluded_boxes=clean_component_payload["excluded_boxes"],
        base_line_points=clean_component_payload["base_line_points"],
        chart_title="RR clean",
        default=None,
        key=f"rr_clean_viewer_{session.session_id}",
    )
    st.caption("Glisse pour d?placer, utilise la molette pour zoomer, double-clique pour r?initialiser et le bouton du composant pour passer en plein ?cran. Les zones denses restent encadr?es et les corrections gardent leur code couleur.")

    if manual_annotations:
        st.caption("Annotations manuelles actives : aucun export clean persistant n'est ?crit tant qu'au moins un flag manuel est pr?sent.")
    elif activer_iteratif and activer_2bis:
        st.caption("Modes it?ratif et 2 bis actifs : aucun export clean persistant n'est ?crit.")
    elif activer_iteratif:
        st.caption("Mode it?ratif actif : aucun export clean persistant n'est ?crit.")
    elif activer_2bis:
        st.caption("Mode 2 bis actif : aucun export clean persistant n'est ?crit.")
    else:
        clean_paths = export_clean_result(repository, session, result, params)
        st.caption(f"Export clean automatique : {clean_paths['rr_clean_relative']} | {clean_paths['fc_clean_relative']}")

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

    if not manual_analysis_table.empty:
        summary_cols = st.columns(3)
        summary_cols[0].metric("Points manuels", str(len(manual_analysis_table)))
        label_auto_counts = manual_analysis_table["label_auto"].fillna("").astype(str).value_counts().to_dict() if "label_auto" in manual_analysis_table.columns else {}
        summary_cols[1].metric("Labels auto distincts", str(len(label_auto_counts)))
        if "t_offset_ms" in manual_analysis_table.columns and not manual_analysis_table["t_offset_ms"].dropna().empty:
            t_min = float(manual_analysis_table["t_offset_ms"].min()) / 60000.0
            t_max = float(manual_analysis_table["t_offset_ms"].max()) / 60000.0
            summary_cols[2].metric("Plage temporelle", f"{t_min:.2f} - {t_max:.2f} min")
        else:
            summary_cols[2].metric("Plage temporelle", "-")
        render_section_label("Points annot?s manuellement")
        st.dataframe(manual_analysis_table, use_container_width=True, height=min(max(180, 44 * (len(manual_analysis_table) + 1)), 340))

    render_section_label("Variables interm?diaires")
    variable_columns = [
        "t_offset_ms", "rr_interval_ms", "source_pipeline_flag", "deco_flag", "label_initial", "label", "label_auto", "manual_flag", "manual_treated_as", "is_manual_annotation", "label_2bis", "label_iteratif", "pass_detected", "run_flag", "run_series_flag",
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
            "t_offset_ms", "rr_interval_ms", "source_pipeline_flag", "deco_flag", "label", "label_auto", "manual_flag", "manual_treated_as", "is_manual_annotation", "label_2bis", "label_iteratif", "pass_detected",
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
            manual_flag=result.cleaned_frame.get("manual_flag", MANUAL_FLAG_OK),
            manual_treated_as=result.cleaned_frame.get("manual_treated_as", ""),
            is_manual_annotation=result.cleaned_frame.get("is_manual_annotation", False),
        )
        clean_columns = [
            "cleaned_index", "rr_interval_ms", "rr_clean_ms", "label", "manual_flag", "manual_treated_as", "is_manual_annotation", "label_2bis", "label_iteratif", "pass_detected",
            "run_flag", "run_series_flag", "deco_flag", "correction_flag", "correction_flag_final", "point_nettoye",
            "zone_dense_artefact", "suspect_zone_dense", "fc_ok", "hrr_ok", "rmssd_ok", "segment_status",
            "quality_segment_label", "dense_region_id", "source_reference",
        ]
        clean_available = [column for column in clean_columns if column in clean_table_source.columns]
        cleaned_table = clean_table_source[clean_available] if show_full_tables else clean_table_source[clean_available].head(700)
        st.dataframe(cleaned_table, use_container_width=True, height=340)



if __name__ == "__main__":
    main()
