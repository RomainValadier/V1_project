from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from polar_app.rr_pipeline import (
    LABEL_A_VERIFIER,
    LABEL_ARTEFACT_ABSOLU,
    LABEL_COURT,
    LABEL_FAUX_BATTEMENT,
    LABEL_GAP_DECO,
    LABEL_LONG,
    LABEL_MANQUE,
    LABEL_OK,
    RRCleaningParams,
    RRCleaningResult,
    _classify_lipponen_candidate,
    analyze_rr_artifacts,
)

PASS_MARKERS = {
    2: ("#f97316", "triangle-up"),
    3: ("#eab308", "diamond"),
    4: ("#84cc16", "star"),
    5: ("#06b6d4", "cross"),
}


@dataclass(slots=True)
class IterativePassSummary:
    pass_number: int
    new_artifacts: int
    dense_points: int
    dense_pct: float
    suspect_points: int
    stop_reason: str = ""


@dataclass(slots=True)
class IterativeLipponenResult:
    result: RRCleaningResult
    final_labels: list[str]
    pass_summaries: list[IterativePassSummary]
    final_pass: int
    stop_reason: str
    base_artifacts: int
    dense_points_last: int
    dense_pct_last: float
    suspect_points_last: int


def _safe_series(frame: pd.DataFrame, column: str, default) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _build_block_ids(df: pd.DataFrame) -> np.ndarray:
    segment_ids = pd.to_numeric(_safe_series(df, "segment_final_id", 0), errors="coerce").fillna(0).astype(int).to_numpy()
    source_flags = _safe_series(df, "source_pipeline_flag", "OK").fillna("OK").astype(str).str.upper().to_numpy()
    labels = _safe_series(df, "label", LABEL_OK).fillna(LABEL_OK).astype(str).to_numpy()
    block_ids = np.full(len(df), -1, dtype=int)
    current_block = -1
    previous_segment = None
    for idx in range(len(df)):
        is_gap = source_flags[idx] == "GAP" or labels[idx] == LABEL_GAP_DECO
        if is_gap:
            previous_segment = None
            continue
        if previous_segment != segment_ids[idx]:
            current_block += 1
            previous_segment = segment_ids[idx]
        block_ids[idx] = current_block
    return block_ids


def _build_block_index_map(block_ids: np.ndarray) -> tuple[dict[int, list[int]], dict[int, int]]:
    block_to_indices: dict[int, list[int]] = {}
    pos_in_block = np.full(len(block_ids), -1, dtype=int)
    for idx, block_id in enumerate(block_ids.tolist()):
        if block_id < 0:
            continue
        bucket = block_to_indices.setdefault(int(block_id), [])
        pos_in_block[idx] = len(bucket)
        bucket.append(idx)
    return block_to_indices, {idx: int(pos_in_block[idx]) for idx in range(len(pos_in_block)) if pos_in_block[idx] >= 0}


def compute_dense_zones(df: pd.DataFrame, window: int = 121, threshold: float = 0.15, artifact_mask: np.ndarray | None = None) -> np.ndarray:
    labels = _safe_series(df, "label", LABEL_OK).fillna(LABEL_OK).astype(str).to_numpy()
    block_ids = _build_block_ids(df)
    block_to_indices, pos_in_block = _build_block_index_map(block_ids)
    dense = np.zeros(len(df), dtype=bool)
    artifact_flags = artifact_mask if artifact_mask is not None else (labels != LABEL_OK)
    half = max(int(window) // 2, 0)
    for block_id, indices in block_to_indices.items():
        block_len = len(indices)
        for idx in indices:
            pos = pos_in_block[idx]
            start = max(0, pos - half)
            end = min(block_len, pos + half + 1)
            window_indices = indices[start:end]
            if not window_indices:
                continue
            artefact_count = int(np.sum(artifact_flags[np.array(window_indices, dtype=int)]))
            dense[idx] = (artefact_count / float(len(window_indices))) >= float(threshold)
    return dense


def _window_indices(indices: list[int], pos: int, window: int) -> list[int]:
    half = max(int(window) // 2, 0)
    start = max(0, pos - half)
    end = min(len(indices), pos + half + 1)
    return indices[start:end]


def _nearest_ok_window(indices: list[int], pos: int, eligible_ok: np.ndarray, initial_radius: int = 5, max_radius: int = 10, target_ok: int = 11) -> list[int]:
    base_start = max(0, pos - initial_radius)
    base_end = min(len(indices), pos + initial_radius + 1)
    candidates = indices[base_start:base_end]
    ok_candidates = [idx for idx in candidates if eligible_ok[idx]]
    if len(ok_candidates) >= target_ok:
        return ok_candidates[:target_ok]
    ext_start = max(0, pos - max_radius)
    ext_end = min(len(indices), pos + max_radius + 1)
    ext_candidates = indices[ext_start:ext_end]
    ok_ext = [idx for idx in ext_candidates if eligible_ok[idx]]
    if len(ok_ext) <= target_ok:
        return ok_ext
    scored = [
        (abs((ext_start + offset) - pos), ext_start + offset, idx)
        for offset, idx in enumerate(ext_candidates)
        if eligible_ok[idx]
    ]
    scored.sort(key=lambda item: (item[0], item[1]))
    return sorted(item[2] for item in scored[:target_ok])


def _compute_iterative_drr(rr: np.ndarray, indices: list[int], eligible_ok: np.ndarray, k_drr_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    drr_iter = np.full(len(rr), np.nan)
    drr_gap = np.full(len(rr), np.nan)
    pred_idx = np.full(len(rr), -1, dtype=int)
    previous_ok = None
    for idx in indices:
        if not eligible_ok[idx]:
            continue
        if previous_ok is None:
            previous_ok = idx
            continue
        gap = idx - previous_ok
        pred_idx[idx] = previous_ok
        if gap <= int(k_drr_max):
            drr_iter[idx] = (rr[idx] - rr[previous_ok]) / float(gap)
            drr_gap[idx] = float(gap)
        previous_ok = idx
    return drr_iter, drr_gap, pred_idx


def _compute_iterative_mrr(rr: np.ndarray, indices: list[int], eligible_ok: np.ndarray, pos_lookup: dict[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    med_local = np.full(len(rr), np.nan)
    mrr_iter = np.full(len(rr), np.nan)
    suspect = np.zeros(len(rr), dtype=bool)
    for idx in indices:
        if not eligible_ok[idx]:
            continue
        pos = pos_lookup[idx]
        ok_window = _nearest_ok_window(indices, pos, eligible_ok)
        if len(ok_window) < 6:
            suspect[idx] = True
            continue
        values = rr[np.array(ok_window, dtype=int)]
        med = float(np.median(values))
        med_local[idx] = med
        delta = rr[idx] - med
        mrr_iter[idx] = (2.0 * delta) if delta < 0 else delta
    return med_local, mrr_iter, suspect


def _compute_qd(values: np.ndarray, floor_ms: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    abs_values = np.abs(finite)
    q1 = float(np.percentile(abs_values, 25))
    q3 = float(np.percentile(abs_values, 75))
    return max((q3 - q1) / 2.0, float(floor_ms))


def _map_rr_clean(cleaned_frame: pd.DataFrame, n_points: int) -> np.ndarray:
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


def _map_cleaned_text(cleaned_frame: pd.DataFrame, n_points: int, column: str, default: str) -> np.ndarray:
    mapped = np.full(n_points, default, dtype=object)
    if cleaned_frame.empty or column not in cleaned_frame.columns:
        return mapped
    grouped = cleaned_frame.groupby(["source_index_start", "source_index_end"], sort=False, dropna=False)
    for (start, end), group in grouped:
        if pd.isna(start) or pd.isna(end):
            continue
        start_i = int(start)
        end_i = int(end)
        value = str(group.iloc[0][column]) if pd.notna(group.iloc[0][column]) else default
        mapped[start_i : end_i + 1] = value
    return mapped


def _aggregate_iterative_label(values: list[str]) -> str:
    non_default = [value for value in values if value and value != "aucun"]
    if not non_default:
        return "aucun"
    unique = list(dict.fromkeys(non_default))
    return unique[0] if len(unique) == 1 else "mixte"


def _attach_iterative_columns(final_result: RRCleaningResult, iterative_analysis: pd.DataFrame) -> RRCleaningResult:
    analysis = final_result.analysis_frame.copy()
    iterative_cols = [
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
    for column in iterative_cols:
        if column in iterative_analysis.columns:
            analysis[column] = iterative_analysis[column].values
    analysis["correction_flag_final"] = _map_cleaned_text(final_result.cleaned_frame, len(analysis), "correction_flag", "ok")
    analysis["rr_clean_ms"] = _map_rr_clean(final_result.cleaned_frame, len(analysis))
    final_result.analysis_frame = analysis

    cleaned = final_result.cleaned_frame.copy()
    if not cleaned.empty:
        label_iter = iterative_analysis.get("label_iteratif", pd.Series("aucun", index=iterative_analysis.index)).astype(str)
        pass_detected = iterative_analysis.get("pass_detected", pd.Series(0, index=iterative_analysis.index)).astype(int)
        dense_flag = iterative_analysis.get("zone_dense_artefact", pd.Series(False, index=iterative_analysis.index)).fillna(False).astype(bool)
        cleaned["label_iteratif"] = cleaned.apply(
            lambda row: _aggregate_iterative_label(label_iter.iloc[int(row["source_index_start"]): int(row["source_index_end"]) + 1].tolist()) if pd.notna(row.get("source_index_start")) and pd.notna(row.get("source_index_end")) else "aucun",
            axis=1,
        )
        cleaned["pass_detected"] = cleaned.apply(
            lambda row: int(pass_detected.iloc[int(row["source_index_start"]): int(row["source_index_end"]) + 1].max()) if pd.notna(row.get("source_index_start")) and pd.notna(row.get("source_index_end")) else 0,
            axis=1,
        )
        cleaned["zone_dense_artefact"] = cleaned.apply(
            lambda row: bool(dense_flag.iloc[int(row["source_index_start"]): int(row["source_index_end"]) + 1].any()) if pd.notna(row.get("source_index_start")) and pd.notna(row.get("source_index_end")) else False,
            axis=1,
        )
        cleaned["correction_flag_final"] = cleaned["correction_flag"].astype(str)
        cleaned["rr_clean_ms"] = cleaned["rr_interval_ms"].astype(float)
    final_result.cleaned_frame = cleaned
    return final_result


def iterative_reclassification(
    df: pd.DataFrame,
    max_iter: int = 3,
    dense_threshold: float = 0.15,
    dense_window: int = 121,
    qd_floor: float = 5.0,
    k_drr_max: int = 5,
    seuil_rendement: int = 3,
    alpha: float = 5.2,
    params: RRCleaningParams | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> IterativeLipponenResult:
    params = params or RRCleaningParams()
    analysis = df.copy().reset_index(drop=True)
    rr = analysis["rr_interval_ms"].astype(float).to_numpy()
    base_labels = analysis.get("label", pd.Series(LABEL_OK, index=analysis.index)).fillna(LABEL_OK).astype(str).tolist()
    current_labels = list(base_labels)
    label_iteratif = np.full(len(analysis), "aucun", dtype=object)
    pass_detected = np.array([1 if label != LABEL_OK else 0 for label in base_labels], dtype=int)
    mrr_local = np.full(len(analysis), np.nan)
    qd_drr_local = np.full(len(analysis), np.nan)
    qd_mrr_local = np.full(len(analysis), np.nan)
    th1_local = np.full(len(analysis), np.nan)
    th2_local = np.full(len(analysis), np.nan)
    drr_iter_local = np.full(len(analysis), np.nan)
    drr_gap_local = np.full(len(analysis), np.nan)
    suspect_zone_dense = np.zeros(len(analysis), dtype=bool)
    zone_dense_artefact = np.zeros(len(analysis), dtype=bool)

    block_ids = _build_block_ids(analysis)
    block_to_indices, pos_lookup = _build_block_index_map(block_ids)
    source_flags = _safe_series(analysis, "source_pipeline_flag", "OK").fillna("OK").astype(str).str.upper().to_numpy()
    correction_raw = _safe_series(analysis, "correction_flag_raw", "ok").fillna("ok").astype(str).to_numpy()
    base_drr_norm = pd.to_numeric(_safe_series(analysis, "dRR_norm", np.nan), errors="coerce").to_numpy(dtype=float)
    non_gap_mask = block_ids >= 0
    reclassifiable_mask = non_gap_mask & (correction_raw != "not_cleaned")

    pass_summaries: list[IterativePassSummary] = []
    stop_reason = "max_iterations"
    dense_points_last = 0
    dense_pct_last = 0.0
    suspect_points_last = 0
    final_pass = 1

    total_passes = max(int(max_iter) - 1, 1)

    for pass_number in range(2, int(max_iter) + 1):
        artifact_mask = np.array([label != LABEL_OK for label in current_labels], dtype=bool)
        artifact_mask |= ~reclassifiable_mask & (source_flags != "GAP") & np.array([label != LABEL_OK for label in current_labels], dtype=bool)
        zone_dense_artefact = compute_dense_zones(analysis.assign(label=current_labels), window=dense_window, threshold=dense_threshold, artifact_mask=artifact_mask)
        dense_points_last = int(zone_dense_artefact.sum())
        dense_pct_last = (100.0 * dense_points_last / float(max(int(non_gap_mask.sum()), 1)))

        eligible_ok = reclassifiable_mask & (~artifact_mask)
        drr_iter = np.full(len(analysis), np.nan)
        drr_gap = np.full(len(analysis), np.nan)
        pred_idx = np.full(len(analysis), -1, dtype=int)
        med_local = np.full(len(analysis), np.nan)
        mrr_iter = np.full(len(analysis), np.nan)
        local_suspect = np.zeros(len(analysis), dtype=bool)

        for indices in block_to_indices.values():
            seg_drr, seg_gap, seg_pred = _compute_iterative_drr(rr, indices, eligible_ok, k_drr_max)
            drr_iter = np.where(np.isfinite(seg_drr), seg_drr, drr_iter)
            drr_gap = np.where(np.isfinite(seg_gap), seg_gap, drr_gap)
            pred_idx = np.where(seg_pred >= 0, seg_pred, pred_idx)
            seg_med, seg_mrr, seg_suspect = _compute_iterative_mrr(rr, indices, eligible_ok, pos_lookup)
            med_local = np.where(np.isfinite(seg_med), seg_med, med_local)
            mrr_iter = np.where(np.isfinite(seg_mrr), seg_mrr, mrr_iter)
            local_suspect |= seg_suspect

        th1_iter = np.full(len(analysis), np.nan)
        th2_iter = np.full(len(analysis), np.nan)
        drr_norm_iter = np.full(len(analysis), np.nan)
        mrr_norm_iter = np.full(len(analysis), np.nan)
        qd_drr = np.full(len(analysis), np.nan)
        qd_mrr = np.full(len(analysis), np.nan)
        new_artifacts_this_pass: list[int] = []

        dense_ok_indices = [idx for idx in range(len(analysis)) if eligible_ok[idx] and zone_dense_artefact[idx]]
        total_dense_ok = len(dense_ok_indices)
        for progress_idx, idx in enumerate(dense_ok_indices, start=1):
            block_indices = block_to_indices.get(int(block_ids[idx]), [])
            if not block_indices:
                continue
            pos = pos_lookup[idx]
            qd_window = _window_indices(block_indices, pos, params.window_th)
            drr_candidates = np.array([drr_iter[j] for j in qd_window if eligible_ok[j] and np.isfinite(drr_iter[j])], dtype=float)
            mrr_candidates = np.array([mrr_iter[j] for j in qd_window if eligible_ok[j] and np.isfinite(mrr_iter[j])], dtype=float)
            if drr_candidates.size < 30 or mrr_candidates.size < 30 or not np.isfinite(med_local[idx]):
                local_suspect[idx] = True
                continue
            qd_drr[idx] = _compute_qd(drr_candidates, qd_floor)
            qd_mrr[idx] = _compute_qd(mrr_candidates, qd_floor)
            th1_iter[idx] = alpha * qd_drr[idx] if np.isfinite(qd_drr[idx]) else np.nan
            th2_iter[idx] = alpha * qd_mrr[idx] if np.isfinite(qd_mrr[idx]) else np.nan
            if not np.isfinite(th1_iter[idx]) or not np.isfinite(th2_iter[idx]) or not np.isfinite(mrr_iter[idx]):
                local_suspect[idx] = True
                continue
            if pred_idx[idx] >= 0 and pred_idx[idx] in new_artifacts_this_pass:
                current_drr_norm = np.nan
            elif np.isfinite(drr_iter[idx]) and np.isfinite(th1_iter[idx]) and th1_iter[idx] != 0:
                current_drr_norm = drr_iter[idx] / th1_iter[idx]
            else:
                current_drr_norm = np.nan
            drr_norm_iter[idx] = current_drr_norm
            mrr_norm_iter[idx] = mrr_iter[idx] / th2_iter[idx] if th2_iter[idx] != 0 else np.nan

            def neighbor_norm(target_idx: int) -> float:
                if target_idx < 0 or target_idx >= len(analysis) or not eligible_ok[target_idx]:
                    return np.nan
                if zone_dense_artefact[target_idx]:
                    return drr_norm_iter[target_idx] if np.isfinite(drr_norm_iter[target_idx]) else np.nan
                return float(base_drr_norm[target_idx]) if np.isfinite(base_drr_norm[target_idx]) else np.nan

            s21_value = current_drr_norm
            neighbor_1 = neighbor_norm(idx + 1)
            neighbor_2 = neighbor_norm(idx + 2)
            if np.isnan(s21_value):
                s22_value = np.nan
            else:
                candidates = [value for value in [neighbor_1, neighbor_2] if not np.isnan(value)]
                s22_value = np.nan if not candidates else (min(candidates) if s21_value >= 0 else max(candidates))
            mrr_norm_value = float(mrr_norm_iter[idx]) if np.isfinite(mrr_norm_iter[idx]) else np.nan
            if np.isnan(s22_value) and (not np.isfinite(mrr_norm_value) or abs(mrr_norm_value) <= params.threshold_mrr):
                local_suspect[idx] = True
                continue
            next_ok_rr = np.nan
            for next_idx in block_indices[pos + 1:]:
                if eligible_ok[next_idx]:
                    next_ok_rr = rr[next_idx]
                    break
            new_label = _classify_lipponen_candidate(
                rr_value=rr[idx],
                med_value=med_local[idx],
                th2_value=th2_iter[idx],
                s21_value=s21_value,
                s22_value=s22_value,
                mrr_norm_value=mrr_norm_value,
                threshold_mrr=params.threshold_mrr,
                next_rr_value=next_ok_rr,
            )
            mrr_local[idx] = mrr_iter[idx]
            qd_drr_local[idx] = qd_drr[idx]
            qd_mrr_local[idx] = qd_mrr[idx]
            th1_local[idx] = th1_iter[idx]
            th2_local[idx] = th2_iter[idx]
            drr_iter_local[idx] = drr_iter[idx]
            drr_gap_local[idx] = drr_gap[idx]
            if new_label != LABEL_OK:
                current_labels[idx] = new_label
                label_iteratif[idx] = f"{new_label}_{pass_number}"
                pass_detected[idx] = pass_number
                new_artifacts_this_pass.append(idx)
            if progress_callback is not None and total_dense_ok:
                base_ratio = (pass_number - 2) / float(total_passes)
                step_ratio = progress_idx / float(total_dense_ok)
                progress_callback(min(0.98, base_ratio + (step_ratio / float(total_passes))))

        suspect_zone_dense |= local_suspect & zone_dense_artefact
        suspect_points_last = int((suspect_zone_dense & zone_dense_artefact).sum())
        new_count = len(new_artifacts_this_pass)
        stop_for_pass = ""
        if new_count == 0:
            stop_for_pass = "convergence"
        elif new_count < int(seuil_rendement):
            stop_for_pass = "rendement_decroissant"
        elif pass_number == int(max_iter):
            stop_for_pass = "max_iterations"
        pass_summaries.append(
            IterativePassSummary(
                pass_number=pass_number,
                new_artifacts=new_count,
                dense_points=dense_points_last,
                dense_pct=dense_pct_last,
                suspect_points=suspect_points_last,
                stop_reason=stop_for_pass,
            )
        )
        print(
            f"Pass {pass_number} : {new_count} nouveaux artefacts detectes | zones denses : {dense_points_last} points ({dense_pct_last:.1f}%) | suspects non classifies : {suspect_points_last}" + (f" | arret : {stop_for_pass}" if stop_for_pass else "")
        )
        final_pass = pass_number
        if stop_for_pass:
            stop_reason = stop_for_pass
            break

    iterative_analysis = analysis.copy()
    iterative_analysis["label"] = pd.Series(current_labels, index=iterative_analysis.index, dtype="object")
    iterative_analysis["label_iteratif"] = pd.Series(label_iteratif, index=iterative_analysis.index, dtype="object")
    iterative_analysis["pass_detected"] = pd.Series(pass_detected, index=iterative_analysis.index, dtype=int)
    iterative_analysis["zone_dense_artefact"] = pd.Series(zone_dense_artefact, index=iterative_analysis.index, dtype=bool)
    iterative_analysis["mrr_local"] = mrr_local
    iterative_analysis["qd_drr_local"] = qd_drr_local
    iterative_analysis["qd_mrr_local"] = qd_mrr_local
    iterative_analysis["th1_local"] = th1_local
    iterative_analysis["th2_local"] = th2_local
    iterative_analysis["drr_iter"] = drr_iter_local
    iterative_analysis["drr_gap"] = drr_gap_local
    iterative_analysis["suspect_zone_dense"] = pd.Series(suspect_zone_dense, index=iterative_analysis.index, dtype=bool)

    final_result = analyze_rr_artifacts(df, params=params, labels_override=current_labels, progress_callback=progress_callback)
    final_result = _attach_iterative_columns(final_result, iterative_analysis)
    if progress_callback is not None:
        progress_callback(1.0)
    return IterativeLipponenResult(
        result=final_result,
        final_labels=current_labels,
        pass_summaries=pass_summaries,
        final_pass=final_pass,
        stop_reason=stop_reason,
        base_artifacts=int(np.sum(np.array(base_labels, dtype=object) != LABEL_OK)),
        dense_points_last=dense_points_last,
        dense_pct_last=dense_pct_last,
        suspect_points_last=suspect_points_last,
    )
