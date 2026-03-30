from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ARTIFACT_LABELS = {
    "artefact_absolu",
    "ectopique",
    "long",
    "court",
    "battement_manque",
    "faux_battement",
}


@dataclass(slots=True)
class RRCleaningParams:
    rr_min_ms: int = 250
    rr_max_ms: int = 1500
    alpha: float = 5.2
    window_th: int = 91
    window_median: int = 11
    threshold_drr: float = 1.0
    threshold_mrr: float = 3.0
    c1: float = 0.13
    c2: float = 0.17
    seuil_alerte: float = 0.05
    seuil_exclusion: float = 0.15
    cluster_consecutive_limit: int = 4
    spline_neighbors: int = 3
    segmentation_gap_break_seconds: int = 10
    absolute_artifact_run_break_count: int = 10
    reliable_restart_count: int = 3

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"parametre": key, "valeur": value} for key, value in asdict(self).items()]
        )


@dataclass(slots=True)
class RRCleaningResult:
    analysis_frame: pd.DataFrame
    cleaned_frame: pd.DataFrame
    ectopic_log_frame: pd.DataFrame
    error_counts: dict[str, int]
    corrected_artifacts: int
    corrected_percentage: float
    long_sequences_count: int
    total_points: int
    cleaned_points: int
    pipeline_flag: str
    consecutive_artifact_flag: str
    cluster_ranges: list[tuple[int, int]]
    cleaning_segments_count: int
    visible_excluded_points: int
    outside_support_points: int


def _deviation_quartile(values: np.ndarray) -> float:
    clean_values = values[~np.isnan(values)]
    if clean_values.size == 0:
        return 0.0
    q1 = float(np.percentile(clean_values, 25))
    q3 = float(np.percentile(clean_values, 75))
    return (q3 - q1) / 2.0


def _rolling_dynamic_threshold(values: np.ndarray, alpha: float, window: int) -> np.ndarray:
    half_window = window // 2
    thresholds = np.zeros(len(values), dtype=float)

    for idx in range(len(values)):
        start = max(0, idx - half_window)
        end = min(len(values), idx + half_window + 1)
        dq = _deviation_quartile(values[start:end])
        thresholds[idx] = alpha * dq

    positive_thresholds = thresholds[thresholds > 0]
    fallback = float(np.median(positive_thresholds)) if positive_thresholds.size else 1.0
    thresholds[thresholds <= 0] = fallback
    return thresholds


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    return series.rolling(window=window, center=True, min_periods=1).median().bfill().ffill().to_numpy()


def _neighbor_value(values: np.ndarray, idx: int) -> float:
    if 0 <= idx < len(values):
        value = values[idx]
        if not np.isnan(value):
            return float(value)
    return np.nan


def _nan_extreme(candidates: list[float], mode: str) -> float:
    valid = [float(value) for value in candidates if not np.isnan(value)]
    if not valid:
        return np.nan
    return max(valid) if mode == "max" else min(valid)


def _compute_cluster_ranges(labels: list[str], strict_threshold: int) -> list[tuple[int, int]]:
    clusters: list[tuple[int, int]] = []
    start = None

    for idx, label in enumerate(labels):
        if label in ARTIFACT_LABELS:
            if start is None:
                start = idx
        else:
            if start is not None and (idx - start) > strict_threshold:
                clusters.append((start, idx - 1))
            start = None

    if start is not None and (len(labels) - start) > strict_threshold:
        clusters.append((start, len(labels) - 1))

    return clusters


def _interpolate_spline_value(
    rr_values: np.ndarray,
    labels: list[str],
    target_idx: int,
    neighbors: int,
    local_median: float,
) -> float:
    healthy_indices = [
        idx for idx, label in enumerate(labels) if label == "normal" and not np.isnan(rr_values[idx])
    ]
    before = [idx for idx in healthy_indices if idx < target_idx][-neighbors:]
    after = [idx for idx in healthy_indices if idx > target_idx][:neighbors]
    support_indices = before + after

    if not before or not after or len(support_indices) < 2:
        return float(local_median)

    x = np.array(support_indices, dtype=float)
    y = rr_values[support_indices].astype(float)

    try:
        interpolator = PchipInterpolator(x, y)
        estimate = float(interpolator(float(target_idx)))
    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
        estimate = float(local_median)

    if np.isnan(estimate) or estimate <= 0:
        return float(local_median)
    return estimate


def _build_empty_result(frame: pd.DataFrame) -> RRCleaningResult:
    empty = frame.copy()
    empty_log = pd.DataFrame(columns=["index", "RR_avant", "RR_j", "RR_apres", "dRR_norm", "mRR_norm"])
    empty_cleaned = pd.DataFrame(
        columns=[
            "cleaned_index",
            "rr_interval_ms",
            "timeline_step_ms",
            "source_reference",
            "origin_label",
            "source_zone_type",
            "cleaning_segment_id",
            "cluster_id",
            "cluster_length",
            "correction_method",
            "source_index",
        ]
    )
    return RRCleaningResult(
        analysis_frame=empty,
        cleaned_frame=empty_cleaned,
        ectopic_log_frame=empty_log,
        error_counts={
            "artefact_absolu": 0,
            "ectopique": 0,
            "long": 0,
            "court": 0,
            "battement_manque": 0,
            "faux_battement": 0,
        },
        corrected_artifacts=0,
        corrected_percentage=0.0,
        long_sequences_count=0,
        total_points=0,
        cleaned_points=0,
        pipeline_flag="OK",
        consecutive_artifact_flag="OK",
        cluster_ranges=[],
        cleaning_segments_count=0,
        visible_excluded_points=0,
        outside_support_points=0,
    )


def _compute_source_step_ms(analysis: pd.DataFrame) -> np.ndarray:
    if "t_offset_ms" in analysis.columns:
        steps = analysis["t_offset_ms"].diff().astype("float64")
        positive_steps = steps[steps > 0]
        fallback = float(positive_steps.median()) if not positive_steps.empty else 1000.0
        steps = steps.fillna(fallback)
        steps = steps.where(steps > 0, fallback)
        return steps.to_numpy(dtype=float)

    rr_series = analysis["rr_interval_ms"].astype("float64")
    positive_rr = rr_series[rr_series > 0]
    fallback = float(positive_rr.median()) if not positive_rr.empty else 1000.0
    steps = rr_series.where(rr_series > 0, fallback).fillna(fallback)
    return steps.to_numpy(dtype=float)

def _has_reliable_run(reliable_mask: np.ndarray, start_idx: int, minimum_length: int) -> bool:
    end_idx = start_idx + minimum_length
    if start_idx < 0 or end_idx > len(reliable_mask):
        return False
    return bool(np.all(reliable_mask[start_idx:end_idx]))


def _classify_preclean_rows(analysis: pd.DataFrame, params: RRCleaningParams) -> pd.DataFrame:
    classified = analysis.copy()
    rr_values = classified["rr_interval_ms"].astype("float64")
    pipeline_flag = classified.get("pipeline_flag", pd.Series("ok", index=classified.index)).fillna("ok").astype(str)

    classified["is_gap_point"] = rr_values.le(0) | pipeline_flag.eq("gap")
    classified["is_post_reconnect"] = pipeline_flag.eq("post_reconnect")
    classified["is_incoherent"] = pd.Series(False, index=classified.index)
    classified["is_absolute_artifact"] = rr_values.gt(0) & (rr_values.lt(params.rr_min_ms) | rr_values.gt(params.rr_max_ms))
    classified["is_reliable_candidate"] = (
        rr_values.gt(0)
        & ~classified["is_gap_point"]
        & ~classified["is_post_reconnect"]
        & ~classified["is_absolute_artifact"]
    )

    classified["preclean_class"] = np.select(
        [
            classified["is_gap_point"],
            classified["is_post_reconnect"],
            classified["is_absolute_artifact"],
            classified["is_reliable_candidate"],
        ],
        ["gap", "post_reconnect", "artefact_absolu", "reliable_candidate"],
        default="hors_segment",
    )
    classified["source_timeline_step_ms"] = _compute_source_step_ms(classified)
    return classified


def _build_cleaning_segments(analysis: pd.DataFrame, params: RRCleaningParams) -> pd.DataFrame:
    segmented = analysis.copy()
    n = len(segmented)
    support_for_cleaning = np.zeros(n, dtype=bool)
    cleaning_segment_id = np.zeros(n, dtype=int)
    segment_break_reason = np.full(n, "", dtype=object)
    source_zone_type = segmented["preclean_class"].astype(object).to_numpy()

    gap_mask = segmented["is_gap_point"].to_numpy(dtype=bool)
    absolute_mask = segmented["is_absolute_artifact"].to_numpy(dtype=bool)
    reliable_mask = segmented["is_reliable_candidate"].to_numpy(dtype=bool)
    post_reconnect_mask = segmented["is_post_reconnect"].to_numpy(dtype=bool)
    step_ms = segmented["source_timeline_step_ms"].to_numpy(dtype=float)

    active_segment_id = 0
    next_segment_id = 0
    idx = 0

    while idx < n:
        if active_segment_id == 0:
            if _has_reliable_run(reliable_mask, idx, params.reliable_restart_count):
                next_segment_id += 1
                active_segment_id = next_segment_id
                support_for_cleaning[idx] = True
                cleaning_segment_id[idx] = active_segment_id
                source_zone_type[idx] = "reliable_support"
                idx += 1
                continue

            if gap_mask[idx]:
                source_zone_type[idx] = "gap_long"
                segment_break_reason[idx] = "gap_long"
            elif absolute_mask[idx]:
                source_zone_type[idx] = "absolute_artifact_run"
                segment_break_reason[idx] = "absolute_artifact_run"
            else:
                source_zone_type[idx] = "post_reconnect_warmup"
                segment_break_reason[idx] = "post_reconnect_warmup"
            idx += 1
            continue

        if gap_mask[idx]:
            run_start = idx
            while idx < n and gap_mask[idx]:
                idx += 1
            run_end = idx - 1
            gap_duration_ms = float(step_ms[run_start : run_end + 1].sum())
            stable_restart = _has_reliable_run(reliable_mask, run_end + 1, params.reliable_restart_count)
            if gap_duration_ms < (params.segmentation_gap_break_seconds * 1000.0) and stable_restart:
                cleaning_segment_id[run_start : run_end + 1] = active_segment_id
                source_zone_type[run_start : run_end + 1] = "micro_gap"
            else:
                source_zone_type[run_start : run_end + 1] = "gap_long"
                segment_break_reason[run_start : run_end + 1] = "gap_long"
                active_segment_id = 0
            continue

        if absolute_mask[idx]:
            run_start = idx
            while idx < n and absolute_mask[idx]:
                idx += 1
            run_end = idx - 1
            run_length = run_end - run_start + 1
            if run_length >= params.absolute_artifact_run_break_count:
                source_zone_type[run_start : run_end + 1] = "absolute_artifact_run"
                segment_break_reason[run_start : run_end + 1] = "absolute_artifact_run"
                active_segment_id = 0
            else:
                cleaning_segment_id[run_start : run_end + 1] = active_segment_id
                source_zone_type[run_start : run_end + 1] = "artefact_absolu_visible"
            continue

        if reliable_mask[idx]:
            support_for_cleaning[idx] = True
            cleaning_segment_id[idx] = active_segment_id
            source_zone_type[idx] = "reliable_support"
        else:
            cleaning_segment_id[idx] = active_segment_id
            if post_reconnect_mask[idx]:
                source_zone_type[idx] = "post_reconnect_warmup"
                segment_break_reason[idx] = "post_reconnect_warmup"
            else:
                source_zone_type[idx] = str(segmented.iloc[idx]["preclean_class"])
        idx += 1

    segmented["support_for_cleaning"] = support_for_cleaning
    segmented["cleaning_segment_id"] = cleaning_segment_id
    segmented["segment_break_reason"] = segment_break_reason
    segmented["source_zone_type"] = source_zone_type
    return segmented


def _analyze_support_segment(
    segment_frame: pd.DataFrame,
    params: RRCleaningParams,
) -> tuple[pd.DataFrame, dict[int, list[dict]], set[int], list[dict]]:
    source_indices = segment_frame.index.to_list()
    rr = segment_frame["rr_interval_ms"].astype("float64").to_numpy()
    n = len(rr)

    minimum_points = (2 * params.spline_neighbors) + 3
    if n < minimum_points:
        analysis = pd.DataFrame(index=segment_frame.index)
        analysis["label"] = ["normal"] * n
        analysis["dRR_ms"] = np.full(n, np.nan)
        analysis["Th1_ms"] = np.full(n, np.nan)
        analysis["dRR_norm"] = np.full(n, np.nan)
        analysis["mediane_locale_ms"] = np.full(n, np.nan)
        analysis["mRR_brut_ms"] = np.full(n, np.nan)
        analysis["mRR_ms"] = np.full(n, np.nan)
        analysis["Th2_ms"] = np.full(n, np.nan)
        analysis["mRR_norm"] = np.full(n, np.nan)
        analysis["S11"] = np.full(n, np.nan)
        analysis["S12"] = np.full(n, np.nan)
        analysis["S21"] = np.full(n, np.nan)
        analysis["S22"] = np.full(n, np.nan)
        analysis["rr_cleaned_primary_ms"] = rr
        analysis["rr_cleaned_secondary_ms"] = np.full(n, np.nan)
        analysis["cleaning_action"] = "keep_segment_too_short"
        analysis["post_correction_status"] = "conserve"
        cleaned_rows = {
            source_indices[position]: [
                {
                    "rr_interval_ms": float(rr[position]),
                    "timeline_step_ms": float(rr[position]),
                    "source_reference": f"{source_indices[position] + 1}",
                    "origin_label": "normal",
                    "source_zone_type": "reliable_support",
                    "correction_method": "keep_segment_too_short",
                    "source_index": source_indices[position],
                }
            ]
            for position in range(n)
        }
        return analysis, cleaned_rows, set(), []

    labels = ["normal"] * n

    drr = np.zeros(n, dtype=float)
    if n > 1:
        drr[1:] = rr[1:] - rr[:-1]
    th1 = _rolling_dynamic_threshold(drr, params.alpha, params.window_th)
    drr_norm = np.divide(drr, th1, out=np.zeros_like(drr), where=th1 != 0)

    median_local = _rolling_median(rr, params.window_median)
    mrr_brut = rr - median_local
    mrr = np.where(mrr_brut < 0, 2 * mrr_brut, mrr_brut)
    th2 = _rolling_dynamic_threshold(mrr, params.alpha, params.window_th)
    mrr_norm = np.divide(mrr, th2, out=np.zeros_like(mrr), where=th2 != 0)

    s11 = drr_norm.copy()
    s12 = np.full(n, np.nan, dtype=float)
    s21 = drr_norm.copy()
    s22 = np.full(n, np.nan, dtype=float)

    for idx in range(n):
        prev_val = _neighbor_value(drr_norm, idx - 1)
        next_val = _neighbor_value(drr_norm, idx + 1)
        if drr_norm[idx] > 0:
            s12[idx] = _nan_extreme([prev_val, next_val], mode="max")
        else:
            s12[idx] = _nan_extreme([prev_val, next_val], mode="min")

        next1 = _neighbor_value(drr_norm, idx + 1)
        next2 = _neighbor_value(drr_norm, idx + 2)
        if drr_norm[idx] >= 0:
            s22[idx] = _nan_extreme([next1, next2], mode="min")
        else:
            s22[idx] = _nan_extreme([next1, next2], mode="max")

    for idx in range(n):
        is_suspect = abs(drr_norm[idx]) > params.threshold_drr or abs(mrr_norm[idx]) > params.threshold_mrr
        if not is_suspect:
            labels[idx] = "normal"
            continue

        pnp = s11[idx] > 1.0 and not np.isnan(s12[idx]) and s12[idx] < (-params.c1 * s11[idx]) - params.c2
        npn = s11[idx] < -1.0 and not np.isnan(s12[idx]) and s12[idx] > (-params.c1 * s11[idx]) + params.c2
        if pnp or npn:
            labels[idx] = "ectopique"
            continue

        condition_long = s21[idx] > 1.0 and not np.isnan(s22[idx]) and s22[idx] < -1.0
        condition_court = s21[idx] < -1.0 and not np.isnan(s22[idx]) and s22[idx] > 1.0
        condition_mrr = abs(mrr_norm[idx]) > params.threshold_mrr

        if not (condition_long or condition_court or condition_mrr):
            labels[idx] = "normal"
            continue

        med = median_local[idx]
        if abs((rr[idx] / 2.0) - med) < th2[idx]:
            labels[idx] = "battement_manque"
        elif idx + 1 < n and abs((rr[idx] + rr[idx + 1]) - med) < th2[idx]:
            labels[idx] = "faux_battement"
        else:
            labels[idx] = "long" if rr[idx] > med else "court"

    analysis = pd.DataFrame(index=segment_frame.index)
    analysis["label"] = labels
    analysis["dRR_ms"] = drr
    analysis["Th1_ms"] = th1
    analysis["dRR_norm"] = drr_norm
    analysis["mediane_locale_ms"] = median_local
    analysis["mRR_brut_ms"] = mrr_brut
    analysis["mRR_ms"] = mrr
    analysis["Th2_ms"] = th2
    analysis["mRR_norm"] = mrr_norm
    analysis["S11"] = s11
    analysis["S12"] = s12
    analysis["S21"] = s21
    analysis["S22"] = s22
    analysis["rr_cleaned_primary_ms"] = np.nan
    analysis["rr_cleaned_secondary_ms"] = np.nan
    analysis["cleaning_action"] = "keep"
    analysis["post_correction_status"] = "conserve"

    cleaned_rows_by_source: dict[int, list[dict]] = {}
    consumed_sources: set[int] = set()
    ectopic_rows: list[dict] = []
    skip_next = False

    for idx in range(n):
        if skip_next:
            skip_next = False
            continue

        label = labels[idx]
        original_rr = float(rr[idx])
        source_idx = source_indices[idx]
        corrected_primary = original_rr
        corrected_secondary = np.nan
        action = "keep"

        if label == "battement_manque":
            corrected_primary = original_rr / 2.0
            corrected_secondary = original_rr / 2.0
            action = "division_en_deux_moitie"
            cleaned_rows_by_source[source_idx] = [
                {
                    "rr_interval_ms": corrected_primary,
                    "timeline_step_ms": corrected_primary,
                    "source_reference": f"{source_idx + 1}a",
                    "origin_label": label,
                    "source_zone_type": "reliable_support",
                    "correction_method": action,
                    "source_index": source_idx,
                },
                {
                    "rr_interval_ms": corrected_secondary,
                    "timeline_step_ms": corrected_secondary,
                    "source_reference": f"{source_idx + 1}b",
                    "origin_label": label,
                    "source_zone_type": "reliable_support",
                    "correction_method": action,
                    "source_index": source_idx,
                },
            ]
        elif label == "faux_battement" and idx + 1 < n:
            corrected_primary = original_rr + float(rr[idx + 1])
            action = "fusion_avec_suivant"
            cleaned_rows_by_source[source_idx] = [
                {
                    "rr_interval_ms": corrected_primary,
                    "timeline_step_ms": corrected_primary,
                    "source_reference": f"{source_idx + 1}+{source_indices[idx + 1] + 1}",
                    "origin_label": label,
                    "source_zone_type": "reliable_support",
                    "correction_method": action,
                    "source_index": source_idx,
                }
            ]
            consumed_sources.add(source_indices[idx + 1])
            analysis.loc[source_indices[idx + 1], "post_correction_status"] = "consomme_par_fusion"
            analysis.loc[source_indices[idx + 1], "cleaning_action"] = "consomme_par_fusion"
            skip_next = True
        elif label in {"ectopique", "long", "court"}:
            corrected_primary = _interpolate_spline_value(
                rr_values=rr,
                labels=labels,
                target_idx=idx,
                neighbors=params.spline_neighbors,
                local_median=median_local[idx],
            )
            action = "interpolation_pchip"
            cleaned_rows_by_source[source_idx] = [
                {
                    "rr_interval_ms": corrected_primary,
                    "timeline_step_ms": corrected_primary,
                    "source_reference": f"{source_idx + 1}",
                    "origin_label": label,
                    "source_zone_type": "reliable_support",
                    "correction_method": action,
                    "source_index": source_idx,
                }
            ]
            if label == "ectopique":
                ectopic_rows.append(
                    {
                        "index": source_idx + 1,
                        "RR_avant": float(rr[idx - 1]) if idx - 1 >= 0 else np.nan,
                        "RR_j": float(rr[idx]),
                        "RR_apres": float(rr[idx + 1]) if idx + 1 < n else np.nan,
                        "dRR_norm": float(drr_norm[idx]),
                        "mRR_norm": float(mrr_norm[idx]),
                    }
                )
        else:
            cleaned_rows_by_source[source_idx] = [
                {
                    "rr_interval_ms": original_rr,
                    "timeline_step_ms": original_rr,
                    "source_reference": f"{source_idx + 1}",
                    "origin_label": label,
                    "source_zone_type": "reliable_support",
                    "correction_method": action,
                    "source_index": source_idx,
                }
            ]

        analysis.loc[source_idx, "rr_cleaned_primary_ms"] = corrected_primary
        analysis.loc[source_idx, "rr_cleaned_secondary_ms"] = corrected_secondary
        analysis.loc[source_idx, "cleaning_action"] = action

    return analysis, cleaned_rows_by_source, consumed_sources, ectopic_rows


def _build_excluded_cleaned_row(
    analysis: pd.DataFrame,
    row_index: int,
    correction_method: str,
    rr_value: float,
) -> dict:
    return {
        "rr_interval_ms": rr_value,
        "timeline_step_ms": float(analysis.loc[row_index, "source_timeline_step_ms"]),
        "source_reference": f"{row_index + 1}",
        "origin_label": str(analysis.loc[row_index, "label"]),
        "source_zone_type": str(analysis.loc[row_index, "source_zone_type"]),
        "cleaning_segment_id": int(analysis.loc[row_index, "cleaning_segment_id"]),
        "correction_method": correction_method,
        "source_index": int(row_index),
    }


def _build_cleaned_frame(
    analysis: pd.DataFrame,
    support_cleaned_map: dict[int, list[dict]],
    consumed_sources: set[int],
) -> pd.DataFrame:
    cleaned_rows: list[dict] = []
    index = 0
    total_points = len(analysis)

    while index < total_points:
        if index in consumed_sources:
            index += 1
            continue

        if bool(analysis.loc[index, "support_for_cleaning"]):
            row_entries = support_cleaned_map.get(index)
            if not row_entries:
                row_entries = [
                    {
                        "rr_interval_ms": float(analysis.loc[index, "rr_interval_ms"]),
                        "timeline_step_ms": float(analysis.loc[index, "rr_interval_ms"]),
                        "source_reference": f"{index + 1}",
                        "origin_label": str(analysis.loc[index, "label"]),
                        "source_zone_type": "reliable_support",
                        "correction_method": str(analysis.loc[index, "cleaning_action"]),
                        "source_index": int(index),
                    }
                ]

            for row_entry in row_entries:
                row_entry = row_entry.copy()
                row_entry["cleaned_index"] = len(cleaned_rows)
                row_entry["cleaning_segment_id"] = int(analysis.loc[index, "cleaning_segment_id"])
                cleaned_rows.append(row_entry)
            index += 1
            continue

        zone_type = str(analysis.loc[index, "source_zone_type"])
        if zone_type == "micro_gap":
            run_start = index
            current_segment_id = int(analysis.loc[index, "cleaning_segment_id"])
            while index < total_points and str(analysis.loc[index, "source_zone_type"]) == "micro_gap" and int(analysis.loc[index, "cleaning_segment_id"]) == current_segment_id:
                index += 1
            run_end = index - 1

            left_candidates = analysis.index[(analysis.index < run_start) & analysis["support_for_cleaning"] & analysis["cleaning_segment_id"].eq(current_segment_id)]
            right_candidates = analysis.index[(analysis.index > run_end) & analysis["support_for_cleaning"] & analysis["cleaning_segment_id"].eq(current_segment_id)]
            left_rr = float(analysis.loc[left_candidates.max(), "rr_interval_ms"]) if len(left_candidates) else np.nan
            right_rr = float(analysis.loc[right_candidates.min(), "rr_interval_ms"]) if len(right_candidates) else np.nan
            run_length = run_end - run_start + 1
            interpolable = not np.isnan(left_rr) and not np.isnan(right_rr)

            for offset, row_index in enumerate(range(run_start, run_end + 1), start=1):
                if interpolable:
                    ratio = offset / float(run_length + 1)
                    rr_value = left_rr + (right_rr - left_rr) * ratio
                    method = "interpolation_micro_gap"
                else:
                    rr_value = np.nan
                    method = "exclusion_micro_gap"
                row = _build_excluded_cleaned_row(analysis, row_index, method, rr_value)
                row["cleaned_index"] = len(cleaned_rows)
                cleaned_rows.append(row)
            continue

        if zone_type in {"gap_long", "post_reconnect_warmup", "absolute_artifact_run", "artefact_absolu_visible"}:
            method = {
                "gap_long": "exclusion_gap_long",
                "post_reconnect_warmup": "exclusion_post_reconnect_warmup",
                "absolute_artifact_run": "exclusion_absolute_artifact_run",
                "artefact_absolu_visible": "exclusion_artefact_absolu",
            }[zone_type]
        else:
            method = f"exclusion_{zone_type}"

        row = _build_excluded_cleaned_row(analysis, index, method, np.nan)
        row["cleaned_index"] = len(cleaned_rows)
        cleaned_rows.append(row)
        index += 1

    cleaned_frame = pd.DataFrame(cleaned_rows)
    if cleaned_frame.empty:
        return cleaned_frame

    cleaned_frame["timeline_step_ms"] = cleaned_frame["timeline_step_ms"].astype("float64")
    return cleaned_frame


def analyze_rr_artifacts(
    frame: pd.DataFrame,
    params: RRCleaningParams | None = None,
) -> RRCleaningResult:
    params = params or RRCleaningParams()

    if frame.empty:
        return _build_empty_result(frame)

    analysis = frame.copy().reset_index(drop=True)
    analysis = _classify_preclean_rows(analysis, params)
    analysis = _build_cleaning_segments(analysis, params)

    analysis["label"] = analysis["preclean_class"]
    analysis.loc[analysis["is_reliable_candidate"], "label"] = "normal"
    analysis["is_artifact"] = False
    analysis["dRR_ms"] = np.nan
    analysis["Th1_ms"] = np.nan
    analysis["dRR_norm"] = np.nan
    analysis["mediane_locale_ms"] = np.nan
    analysis["mRR_brut_ms"] = np.nan
    analysis["mRR_ms"] = np.nan
    analysis["Th2_ms"] = np.nan
    analysis["mRR_norm"] = np.nan
    analysis["S11"] = np.nan
    analysis["S12"] = np.nan
    analysis["S21"] = np.nan
    analysis["S22"] = np.nan
    analysis["post_correction_status"] = np.where(analysis["support_for_cleaning"], "conserve", "hors_support")
    analysis["rr_cleaned_primary_ms"] = np.nan
    analysis["rr_cleaned_secondary_ms"] = np.nan
    analysis["cleaning_action"] = np.where(analysis["support_for_cleaning"], "keep", "skip_hors_support")
    analysis.loc[analysis["source_zone_type"] == "artefact_absolu_visible", "post_correction_status"] = "exclu"
    analysis.loc[analysis["source_zone_type"] == "gap_long", "post_correction_status"] = "exclu"
    analysis.loc[analysis["source_zone_type"] == "absolute_artifact_run", "post_correction_status"] = "exclu"
    analysis.loc[analysis["source_zone_type"] == "post_reconnect_warmup", "post_correction_status"] = "hors_support"

    support_cleaned_map: dict[int, list[dict]] = {}
    consumed_sources: set[int] = set()
    ectopic_rows: list[dict] = []

    for segment_id in sorted(analysis.loc[analysis["support_for_cleaning"], "cleaning_segment_id"].unique()):
        segment_rows = analysis.index[
            analysis["support_for_cleaning"] & analysis["cleaning_segment_id"].eq(int(segment_id))
        ]
        if len(segment_rows) == 0:
            continue
        segment_analysis, segment_cleaned_map, segment_consumed, segment_ectopic_rows = _analyze_support_segment(
            analysis.loc[segment_rows],
            params,
        )
        support_cleaned_map.update(segment_cleaned_map)
        consumed_sources.update(segment_consumed)
        ectopic_rows.extend(segment_ectopic_rows)
        for column in segment_analysis.columns:
            analysis.loc[segment_analysis.index, column] = segment_analysis[column]

    analysis["is_artifact"] = analysis["label"].isin(ARTIFACT_LABELS)
    analysis["correction_method"] = analysis["cleaning_action"]
    analysis["rr_cleaned_repr"] = analysis.apply(
        lambda row: (
            f"{row['rr_cleaned_primary_ms']:.1f} | {row['rr_cleaned_secondary_ms']:.1f}"
            if pd.notna(row["rr_cleaned_secondary_ms"])
            else (f"{row['rr_cleaned_primary_ms']:.1f}" if pd.notna(row["rr_cleaned_primary_ms"]) else "-")
        ),
        axis=1,
    )

    labels = analysis["label"].astype(str).tolist()
    cluster_ranges = _compute_cluster_ranges(labels, params.cluster_consecutive_limit)
    long_sequences_count = len(cluster_ranges)
    consecutive_artifact_flag = "ALERTE_CLUSTER" if cluster_ranges else "OK"

    cluster_id_by_index = [0] * len(analysis)
    cluster_length_by_id: dict[int, int] = {}
    for cluster_id, (start, end) in enumerate(cluster_ranges, start=1):
        cluster_length = end - start + 1
        cluster_length_by_id[cluster_id] = cluster_length
        for idx in range(start, end + 1):
            cluster_id_by_index[idx] = cluster_id

    analysis["cluster_id"] = cluster_id_by_index
    analysis["cluster_length"] = [cluster_length_by_id.get(cluster_id, 0) for cluster_id in cluster_id_by_index]

    cleaned_frame = _build_cleaned_frame(analysis, support_cleaned_map, consumed_sources)
    if not cleaned_frame.empty:
        cleaned_frame["cluster_id"] = cleaned_frame["source_index"].map(lambda idx: cluster_id_by_index[int(idx)] if int(idx) < len(cluster_id_by_index) else 0)
        cleaned_frame["cluster_length"] = cleaned_frame["cluster_id"].map(lambda cluster_id: cluster_length_by_id.get(int(cluster_id), 0))
        cleaned_frame = cleaned_frame[
            [
                "cleaned_index",
                "rr_interval_ms",
                "timeline_step_ms",
                "source_reference",
                "origin_label",
                "source_zone_type",
                "cleaning_segment_id",
                "cluster_id",
                "cluster_length",
                "correction_method",
                "source_index",
            ]
        ]

    error_order = ["artefact_absolu", "ectopique", "long", "court", "battement_manque", "faux_battement"]
    error_counts = {label: int(analysis["label"].eq(label).sum()) for label in error_order}

    corrected_artifacts = int(analysis["label"].isin({"ectopique", "long", "court", "battement_manque", "faux_battement"}).sum())
    visible_excluded_points = int(cleaned_frame["rr_interval_ms"].isna().sum()) if not cleaned_frame.empty else 0
    outside_support_points = int((~analysis["support_for_cleaning"]).sum())
    total_points = len(analysis)
    issue_points = corrected_artifacts + visible_excluded_points + outside_support_points
    corrected_percentage_ratio = (issue_points / total_points) if total_points else 0.0

    pipeline_flag = "OK"
    if corrected_percentage_ratio > params.seuil_exclusion:
        pipeline_flag = "EXCLUSION"
    elif corrected_percentage_ratio > params.seuil_alerte:
        pipeline_flag = "ALERTE"

    return RRCleaningResult(
        analysis_frame=analysis,
        cleaned_frame=cleaned_frame,
        ectopic_log_frame=pd.DataFrame(ectopic_rows),
        error_counts=error_counts,
        corrected_artifacts=corrected_artifacts,
        corrected_percentage=corrected_percentage_ratio * 100.0,
        long_sequences_count=long_sequences_count,
        total_points=total_points,
        cleaned_points=int(cleaned_frame["rr_interval_ms"].notna().sum()) if not cleaned_frame.empty else 0,
        pipeline_flag=pipeline_flag,
        consecutive_artifact_flag=consecutive_artifact_flag,
        cluster_ranges=cluster_ranges,
        cleaning_segments_count=int(analysis["cleaning_segment_id"].max()) if not analysis.empty else 0,
        visible_excluded_points=visible_excluded_points,
        outside_support_points=outside_support_points,
    )
