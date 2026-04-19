from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

IMPORT_FLAG_GAP = "GAP"
IMPORT_FLAG_POST_RECONNECT_COURT = "POST_RECONNECT_COURT"
IMPORT_FLAG_POST_RECONNECT_MOYEN = "POST_RECONNECT_MOYEN"
IMPORT_FLAG_POST_RECONNECT_LONG = "POST_RECONNECT_LONG"
IMPORT_FLAG_POST_RECONNECT_TRES_LONG = "POST_RECONNECT_TRES_LONG"

DECO_OK = "ok"
DECO_GAP_COURT = "gap_deco_court"
DECO_GAP_LONG = "gap_deco_long"
DECO_POST_RECONNECT_COURT = "post_reconnect_court"
DECO_POST_RECONNECT_LONG = "post_reconnect_long"
DECO_FLAG_ORDER = [DECO_OK, DECO_GAP_COURT, DECO_GAP_LONG, DECO_POST_RECONNECT_COURT, DECO_POST_RECONNECT_LONG]

LABEL_A_VERIFIER = "a_verifier"
LABEL_OK = "ok"
LABEL_GAP_DECO = "gap_deco"
LABEL_ARTEFACT_ABSOLU = "artefact_absolu"
LABEL_MANQUE = "manque"
LABEL_FAUX_BATTEMENT = "faux_battement"
LABEL_LONG = "long"
LABEL_COURT = "court"
ARTIFACT_LABELS = {LABEL_ARTEFACT_ABSOLU, LABEL_MANQUE, LABEL_FAUX_BATTEMENT, LABEL_LONG, LABEL_COURT}
LABEL_ORDER = [LABEL_OK, LABEL_GAP_DECO, LABEL_ARTEFACT_ABSOLU, LABEL_MANQUE, LABEL_FAUX_BATTEMENT, LABEL_LONG, LABEL_COURT]

RUN_OK = "ok"
RUN_NON_TRAITE = "non_traite"
RUN_ARTEFACT_UNIQUE = "artefact_unique"
RUN_COURT = "run_court"
RUN_MOYEN = "run_moyen"
RUN_LONG = "run_long"
RUN_SERIE = "run_serie"
RUN_GAP_DECO = "gap_deco"
RUN_GAP_ARTEFACT = "gap_artefact"
RUN_POST_RECONNECT_DECO = "post_reconnect_deco"
RUN_POST_RECONNECT_ARTEFACT = "post_reconnect_artefact"
POST_RECONNECT_RUN_FLAGS = {RUN_POST_RECONNECT_DECO, RUN_POST_RECONNECT_ARTEFACT}
EXCLUDED_POST_RECONNECT_RUN_FLAGS = {RUN_POST_RECONNECT_DECO}
RUN_FLAG_ORDER = [RUN_OK, RUN_ARTEFACT_UNIQUE, RUN_COURT, RUN_MOYEN, RUN_LONG, RUN_GAP_DECO, RUN_GAP_ARTEFACT, RUN_POST_RECONNECT_DECO, RUN_POST_RECONNECT_ARTEFACT, RUN_NON_TRAITE]
RUN_SERIES_FLAG_ORDER = [RUN_OK, RUN_SERIE]

CORRECTION_OK = "ok"
CORRECTION_DIVISION = "division"
CORRECTION_FUSION = "fusion"
CORRECTION_INTERPOLATION_PCHIP = "interpolation_pchip"
CORRECTION_INTERPOLATION_LINEAIRE = "interpolation_lineaire"
CORRECTION_NOT_CLEANED = "not_cleaned"
CORRECTION_FLAG_ORDER = [CORRECTION_OK, CORRECTION_DIVISION, CORRECTION_FUSION, CORRECTION_INTERPOLATION_PCHIP, CORRECTION_INTERPOLATION_LINEAIRE, CORRECTION_NOT_CLEANED]

LABEL_2BIS_NONE = "aucun"
LABEL_2BIS_A = "2bis_A"
LABEL_2BIS_B = "2bis_B"

QUALITY_OK = "OK"
QUALITY_ALERT = "ALERTE"
QUALITY_EXCLUSION = "EXCLUSION"
BREAK_TYPE_DECO = "CASSURE_DECO"
BREAK_TYPE_ARTEFACT = "CASSURE_ARTEFACT"

RAW_ARTIFACT_COLORS = {
    LABEL_ARTEFACT_ABSOLU: "#dc2626",
    LABEL_MANQUE: "#16a34a",
    LABEL_FAUX_BATTEMENT: "#f59e0b",
    LABEL_LONG: "#2563eb",
    LABEL_COURT: "#0ea5e9",
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
    seuil_run_court_max: int = 6
    seuil_run_moyen_max: int = 15
    seuil_deco_interpoler: int = 15
    window_densite: int = 20
    seuil_densite: float = 0.50
    y_court: int = 5
    tolerance_court: int = 2
    y_moyen: int = 6
    tolerance_moyen: int = 3
    chauffe_gap_court: int = 5
    chauffe_gap_long: int = 8
    seuil_gap_duree_ms: int = 15000
    seuil_qualite_court: float = 0.05
    seuil_qualite_long: float = 0.15
    duree_seuil_segment_ms: int = 120000

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"parametre": k, "valeur": v} for k, v in asdict(self).items()])

@dataclass(slots=True)
class RRCleaningResult:
    analysis_frame: pd.DataFrame
    cleaned_frame: pd.DataFrame
    segments_frame: pd.DataFrame
    quality_segments_frame: pd.DataFrame
    breaks_frame: pd.DataFrame
    dense_regions_frame: pd.DataFrame
    label_counts: dict[str, int]
    run_flag_counts: dict[str, int]
    run_series_flag_counts: dict[str, int]
    deco_flag_counts: dict[str, int]
    correction_counts: dict[str, int]
    total_points: int
    cleaned_points: int
    ok_rr_total: int
    non_viable_rr_total: int
    n_cassures_total: int
    n_cassures_deco: int
    n_cassures_artefact: int
    n_zones_denses: int
    dense_points_count: int
    n_segments_actifs: int
    n_segments_exclus: int
    correction_rate_outside_breaks: float
    global_non_ok_rate: float
    global_quality_label: str

def _empty_result(frame: pd.DataFrame) -> RRCleaningResult:
    return RRCleaningResult(frame.copy(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}, {}, {}, {}, int(len(frame)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, QUALITY_OK)

def _source_flags(frame: pd.DataFrame) -> pd.Series:
    if "pipeline_flag" not in frame.columns:
        return pd.Series("OK", index=frame.index, dtype="object")
    return frame["pipeline_flag"].fillna("OK").astype(str).str.upper()

def _source_step_ms(frame: pd.DataFrame) -> np.ndarray:
    if "t_offset_ms" in frame.columns:
        diff = frame["t_offset_ms"].astype("float64").diff()
        positive = diff[diff > 0]
        fallback = float(positive.median()) if not positive.empty else 1000.0
        return diff.fillna(fallback).where(diff > 0, fallback).to_numpy(dtype=float)
    rr = frame["rr_interval_ms"].astype("float64")
    positive = rr[rr > 0]
    fallback = float(positive.median()) if not positive.empty else 1000.0
    return rr.where(rr > 0, fallback).fillna(fallback).to_numpy(dtype=float)

def _ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            out.append((start, idx - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out

def _counts(series: pd.Series, order: list[str]) -> dict[str, int]:
    values = series.fillna("").astype(str).value_counts().to_dict()
    return {key: int(values.get(key, 0)) for key in order if int(values.get(key, 0)) > 0}

def _quality(rate: float, low: float = 0.05, high: float = 0.15) -> str:
    if rate > high:
        return QUALITY_EXCLUSION
    if rate > low:
        return QUALITY_ALERT
    return QUALITY_OK


def _emit_progress(progress_callback: Callable[[float], None] | None, value: float) -> None:
    if progress_callback is None:
        return
    progress_callback(min(max(float(value), 0.0), 1.0))


def _wrap_progress_callback(progress_callback: Callable[[float], None] | None) -> Callable[[float], None]:
    last = {"value": 0.0}

    def emit(value: float) -> None:
        bounded = min(max(float(value), 0.0), 1.0)
        if bounded < last["value"]:
            bounded = last["value"]
        last["value"] = bounded
        _emit_progress(progress_callback, bounded)

    return emit


def _progress_chunk(total: int) -> int:
    if total <= 0:
        return 1
    return max(500, total // 50)


def _emit_stage_progress(progress_callback: Callable[[float], None] | None, progress_start: float, progress_end: float, current: int, total: int) -> None:
    if total <= 0:
        _emit_progress(progress_callback, progress_end)
        return
    ratio = min(max(float(current) / float(total), 0.0), 1.0)
    _emit_progress(progress_callback, progress_start + ((progress_end - progress_start) * ratio))

def _rolling_threshold(values: np.ndarray, window: int, alpha: float) -> np.ndarray:
    out = np.ones(len(values), dtype=float)
    half = window // 2
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        sample = values[start:end]
        sample = sample[~np.isnan(sample)]
        if sample.size == 0:
            out[idx] = 1.0
            continue
        q1 = float(np.percentile(sample, 25))
        q3 = float(np.percentile(sample, 75))
        qd = (q3 - q1) / 2.0
        out[idx] = alpha * qd if qd > 0 else 1.0
    return out

def _neighbor(values: np.ndarray, idx: int) -> float:
    if 0 <= idx < len(values) and not np.isnan(values[idx]):
        return float(values[idx])
    return np.nan

def _extreme(values: list[float], mode: str) -> float:
    valid = [float(v) for v in values if not np.isnan(v)]
    if not valid:
        return np.nan
    return max(valid) if mode == "max" else min(valid)

def _classify_lipponen_candidate(
    rr_value: float,
    med_value: float,
    th2_value: float,
    s21_value: float,
    s22_value: float,
    mrr_norm_value: float,
    threshold_mrr: float,
    next_rr_value: float = np.nan,
) -> str:
    condition_long = s21_value > 1.0 and not np.isnan(s22_value) and s22_value < -1.0
    condition_court = s21_value < -1.0 and not np.isnan(s22_value) and s22_value > 1.0
    condition_mrr = abs(mrr_norm_value) > threshold_mrr
    if not (condition_long or condition_court or condition_mrr):
        return LABEL_OK
    if abs((rr_value / 2.0) - med_value) < th2_value:
        return LABEL_MANQUE
    if np.isfinite(next_rr_value) and abs((rr_value + next_rr_value) - med_value) < th2_value:
        return LABEL_FAUX_BATTEMENT
    return LABEL_LONG if rr_value > med_value else LABEL_COURT

def _gap_duration_ms(frame: pd.DataFrame, step_ms: np.ndarray, start: int, end: int) -> float:
    if "timestamp" in frame.columns:
        ts = pd.to_datetime(frame["timestamp"], errors="coerce")
        left = start - 1
        right = end + 1
        if left >= 0 and right < len(frame) and pd.notna(ts.iloc[left]) and pd.notna(ts.iloc[right]):
            return max(float((ts.iloc[right] - ts.iloc[left]).total_seconds() * 1000.0), float(step_ms[start : end + 1].sum()))
    return float(step_ms[start : end + 1].sum())

def _build_deco_flags(frame: pd.DataFrame, params: RRCleaningParams) -> tuple[pd.Series, pd.DataFrame]:
    source_flags = _source_flags(frame)
    rr = frame["rr_interval_ms"].astype("float64")
    step_ms = _source_step_ms(frame)
    deco = pd.Series(DECO_OK, index=frame.index, dtype="object")
    rows: list[dict] = []
    gap_mask = source_flags.eq(IMPORT_FLAG_GAP) | rr.fillna(0).le(0)
    for start, end in _ranges(gap_mask.to_numpy(dtype=bool)):
        length = end - start + 1
        gap_flag = DECO_GAP_COURT if length <= params.seuil_deco_interpoler else DECO_GAP_LONG
        deco.iloc[start : end + 1] = gap_flag
        duration_ms = _gap_duration_ms(frame, step_ms, start, end)
        rows.append({"break_type": BREAK_TYPE_DECO, "index_debut": int(start), "index_fin": int(end), "longueur_battements": int(length), "duree_ms": round(duration_ms, 3), "flag_source": gap_flag})
        if gap_flag == DECO_GAP_LONG:
            warmup = params.chauffe_gap_court if duration_ms < params.seuil_gap_duree_ms else params.chauffe_gap_long
            warm_flag = DECO_POST_RECONNECT_COURT if duration_ms < params.seuil_gap_duree_ms else DECO_POST_RECONNECT_LONG
            next_valid = [idx for idx in range(end + 1, len(frame)) if not gap_mask.iloc[idx]]
            for idx in next_valid[:warmup]:
                deco.iloc[idx] = warm_flag
    return deco, pd.DataFrame(rows)

def _build_initial_segments(analysis: pd.DataFrame, params: RRCleaningParams) -> pd.DataFrame:
    step_ms = analysis["source_timeline_step_ms"].to_numpy(dtype=float)
    ts = pd.to_datetime(analysis.get("timestamp"), errors="coerce") if "timestamp" in analysis.columns else pd.Series(pd.NaT, index=analysis.index)
    rows: list[dict] = []
    seg_id = 0
    idx = 0
    while idx < len(analysis):
        if analysis.loc[idx, "deco_flag"] == DECO_GAP_LONG:
            idx += 1
            continue
        start = idx
        while idx + 1 < len(analysis) and analysis.loc[idx + 1, "deco_flag"] != DECO_GAP_LONG:
            idx += 1
        end = idx
        seg_id += 1
        if pd.notna(ts.iloc[start]) and pd.notna(ts.iloc[end]):
            duration_ms = float((ts.iloc[end] - ts.iloc[start]).total_seconds() * 1000.0)
        else:
            duration_ms = float(step_ms[start : end + 1].sum())
        rows.append({
            "segment_id": seg_id,
            "index_debut": int(start),
            "index_fin": int(end),
            "duree_ms": round(duration_ms, 3),
            "n_battements": int(end - start + 1),
            "seuil_inclusion": params.seuil_qualite_court if duration_ms < params.duree_seuil_segment_ms else params.seuil_qualite_long,
            "statut_segment": "actif",
        })
        idx += 1
    return pd.DataFrame(rows)

def _classify_labels(analysis: pd.DataFrame, params: RRCleaningParams, progress_callback: Callable[[float], None] | None = None, progress_start: float = 0.0, progress_end: float = 1.0) -> pd.DataFrame:
    out = analysis.copy()
    labels = out["label_initial"].astype(str).tolist()
    rr = out["rr_interval_ms"].astype("float64").to_numpy()
    valid_indices = [idx for idx, label in enumerate(labels) if label != LABEL_GAP_DECO]
    drr_raw = np.zeros(len(out), dtype=float)
    th1_raw = np.full(len(out), np.nan)
    drr_norm_raw = np.zeros(len(out), dtype=float)
    med_raw = np.full(len(out), np.nan)
    mrr_brut_raw = np.full(len(out), np.nan)
    mrr_raw = np.full(len(out), np.nan)
    th2_raw = np.full(len(out), np.nan)
    mrr_norm_raw = np.zeros(len(out), dtype=float)
    s21_raw = np.full(len(out), np.nan)
    s22_raw = np.full(len(out), np.nan)
    if valid_indices:
        chunk = _progress_chunk(len(valid_indices))
        valid_rr = rr[valid_indices].astype(float)
        drr = np.zeros(len(valid_rr), dtype=float)
        if len(valid_rr) > 1:
            drr[1:] = valid_rr[1:] - valid_rr[:-1]
        th1 = _rolling_threshold(drr, params.window_th, params.alpha)
        drr_norm = np.divide(drr, th1, out=np.zeros_like(drr), where=th1 != 0)
        med_local = pd.Series(valid_rr, dtype="float64").rolling(window=params.window_median, center=True, min_periods=1).median().bfill().ffill().to_numpy()
        mrr_brut = valid_rr - med_local
        mrr = np.where(mrr_brut < 0, 2.0 * mrr_brut, mrr_brut)
        th2 = _rolling_threshold(mrr, params.window_th, params.alpha)
        mrr_norm = np.divide(mrr, th2, out=np.zeros_like(mrr), where=th2 != 0)
        s21 = drr_norm.copy()
        s22 = np.full(len(valid_rr), np.nan)
        for pos in range(len(valid_rr)):
            s22[pos] = _extreme([_neighbor(drr_norm, pos + 1), _neighbor(drr_norm, pos + 2)], "min" if drr_norm[pos] >= 0 else "max")
        valid_pos = {raw_idx: pos for pos, raw_idx in enumerate(valid_indices)}
        for pos, raw_idx in enumerate(valid_indices):
            drr_raw[raw_idx] = drr[pos]
            th1_raw[raw_idx] = th1[pos]
            drr_norm_raw[raw_idx] = drr_norm[pos]
            med_raw[raw_idx] = med_local[pos]
            mrr_brut_raw[raw_idx] = mrr_brut[pos]
            mrr_raw[raw_idx] = mrr[pos]
            th2_raw[raw_idx] = th2[pos]
            mrr_norm_raw[raw_idx] = mrr_norm[pos]
            s21_raw[raw_idx] = s21[pos]
            s22_raw[raw_idx] = s22[pos]
        for progress_idx, raw_idx in enumerate(valid_indices, start=1):
            if labels[raw_idx] in {LABEL_GAP_DECO, LABEL_ARTEFACT_ABSOLU} or labels[raw_idx] != LABEL_A_VERIFIER:
                continue
            pos = valid_pos[raw_idx]
            if abs(drr_norm[pos]) <= params.threshold_drr and abs(mrr_norm[pos]) <= params.threshold_mrr:
                labels[raw_idx] = LABEL_OK
                continue
            next_rr_value = np.nan
            if raw_idx + 1 < len(out) and labels[raw_idx + 1] != LABEL_GAP_DECO:
                next_rr_value = rr[raw_idx + 1]
            labels[raw_idx] = _classify_lipponen_candidate(
                rr_value=rr[raw_idx],
                med_value=med_local[pos],
                th2_value=th2[pos],
                s21_value=s21[pos],
                s22_value=s22[pos],
                mrr_norm_value=mrr_norm[pos],
                threshold_mrr=params.threshold_mrr,
                next_rr_value=next_rr_value,
            )
            if progress_idx % chunk == 0 or progress_idx == len(valid_indices):
                _emit_stage_progress(progress_callback, progress_start, progress_end, progress_idx, len(valid_indices))
    _emit_progress(progress_callback, progress_end)
    out["label"] = labels
    out["dRR_ms"] = drr_raw
    out["Th1_ms"] = th1_raw
    out["dRR_norm"] = drr_norm_raw
    out["med_locale_ms"] = med_raw
    out["mRR_brut_ms"] = mrr_brut_raw
    out["mRR_ms"] = mrr_raw
    out["Th2_ms"] = th2_raw
    out["mRR_norm"] = mrr_norm_raw
    out["S21"] = s21_raw
    out["S22"] = s22_raw
    return out

def _apply_segment_inclusion(analysis: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    out = segments.copy()
    for row in out.itertuples(index=False):
        segment = analysis.loc[int(row.index_debut) : int(row.index_fin)]
        non_gap = segment.loc[segment["label"].ne(LABEL_GAP_DECO)]
        n_total = int(len(non_gap))
        n_art = int(non_gap["label"].isin(ARTIFACT_LABELS).sum())
        rate = (n_art / n_total) if n_total else np.nan
        status = "exclu" if np.isnan(rate) or rate > float(row.seuil_inclusion) else "actif"
        out.loc[out["segment_id"].eq(row.segment_id), "statut_segment"] = status
        out.loc[out["segment_id"].eq(row.segment_id), "n_total_analyse"] = n_total
        out.loc[out["segment_id"].eq(row.segment_id), "n_artefacts_segment"] = n_art
        out.loc[out["segment_id"].eq(row.segment_id), "taux_artefact_segment"] = None if np.isnan(rate) else round(float(rate), 4)
    return out

def _initial_run_flags(analysis: pd.DataFrame) -> pd.Series:
    run_flags = pd.Series(RUN_NON_TRAITE, index=analysis.index, dtype="object")
    run_flags.loc[analysis["deco_flag"].isin([DECO_GAP_COURT, DECO_GAP_LONG])] = RUN_GAP_DECO
    run_flags.loc[analysis["deco_flag"].isin([DECO_POST_RECONNECT_COURT, DECO_POST_RECONNECT_LONG])] = RUN_POST_RECONNECT_DECO
    run_flags.loc[(analysis["label"] == LABEL_OK) & analysis["deco_flag"].eq(DECO_OK)] = RUN_OK
    return run_flags

def _assign_run_flags(analysis: pd.DataFrame, segments: pd.DataFrame, params: RRCleaningParams, progress_callback: Callable[[float], None] | None = None, progress_start: float = 0.0, progress_end: float = 1.0) -> pd.DataFrame:
    out = analysis.copy()
    out["run_flag"] = _initial_run_flags(out)
    out["run_series_flag"] = RUN_OK
    out["_dense_candidate"] = False
    active_ids = segments.loc[segments["statut_segment"].eq("actif"), "segment_id"].tolist()
    total_valid = 0
    for segment_id in active_ids:
        segment_mask = out["initial_segment_id"].eq(segment_id)
        total_valid += int((segment_mask & out["deco_flag"].eq(DECO_OK)).sum())
    processed_valid = 0
    chunk = _progress_chunk(total_valid)
    for segment_id in active_ids:
        segment_mask = out["initial_segment_id"].eq(segment_id)
        candidate_mask = segment_mask & ~out["label"].isin([LABEL_OK, LABEL_GAP_DECO])
        for start, end in _ranges(candidate_mask.to_numpy(dtype=bool)):
            length = end - start + 1
            if length == 1:
                flag = RUN_ARTEFACT_UNIQUE
            elif length <= params.seuil_run_court_max:
                flag = RUN_COURT
            elif length <= params.seuil_run_moyen_max:
                flag = RUN_MOYEN
            else:
                flag = RUN_LONG
            out.loc[start:end, "run_flag"] = flag
        valid_indices = out.index[segment_mask & out["deco_flag"].eq(DECO_OK)].tolist()
        if not valid_indices:
            continue
        half = params.window_densite // 2
        for pos in range(len(valid_indices)):
            start_pos = max(0, pos - half)
            end_pos = min(len(valid_indices), pos + half + 1)
            window_indices = valid_indices[start_pos:end_pos]
            if not window_indices:
                continue
            density = float(out.loc[window_indices, "label"].isin(ARTIFACT_LABELS).sum()) / float(len(window_indices))
            if density > params.seuil_densite:
                out.loc[window_indices, "_dense_candidate"] = True
            processed_valid += 1
            if total_valid and (processed_valid % chunk == 0 or processed_valid == total_valid):
                _emit_stage_progress(progress_callback, progress_start, progress_end, processed_valid, total_valid)
    out.loc[out["_dense_candidate"], "run_series_flag"] = RUN_SERIE
    _emit_progress(progress_callback, progress_end)
    return out


def _mark_dense_regions(analysis: pd.DataFrame) -> pd.DataFrame:
    out = analysis.copy()
    dense_mask = out.get("_dense_candidate", pd.Series(False, index=out.index)).fillna(False).to_numpy(dtype=bool)
    dense_ids = np.zeros(len(out), dtype=int)
    region_id = 0
    for start, end in _ranges(dense_mask):
        region_id += 1
        dense_ids[start : end + 1] = region_id
    out["dense_region_id"] = dense_ids
    if "_dense_candidate" in out.columns:
        out = out.drop(columns=["_dense_candidate"])
    return out


def _build_dense_regions_frame(analysis: pd.DataFrame, cleaned_frame: pd.DataFrame) -> pd.DataFrame:
    if analysis.empty or "dense_region_id" not in analysis.columns:
        return pd.DataFrame()
    dense = analysis.loc[analysis["dense_region_id"].gt(0)].copy()
    if dense.empty:
        return pd.DataFrame()
    if "t_offset_ms" in dense.columns:
        dense["t_min"] = dense["t_offset_ms"].astype("float64") / 60000.0
    else:
        dense["t_min"] = dense["source_timeline_step_ms"].astype("float64").cumsum() / 60000.0
    rows: list[dict[str, object]] = []
    for dense_region_id, group in dense.groupby("dense_region_id", sort=True):
        cleaned_group = cleaned_frame.loc[cleaned_frame.get("dense_region_id", pd.Series(0, index=cleaned_frame.index)).eq(int(dense_region_id))].copy() if not cleaned_frame.empty else pd.DataFrame()
        correction_values = []
        if not cleaned_group.empty and "correction_flag" in cleaned_group.columns:
            correction_values = sorted({value for value in cleaned_group["correction_flag"].dropna().astype(str) if value not in {CORRECTION_OK, CORRECTION_NOT_CLEANED}})
        segment_values = sorted({int(value) for value in group["segment_final_id"].dropna().astype(int) if int(value) > 0}) if "segment_final_id" in group.columns else []
        n_points = int(len(group))
        n_artifacts = int(group["label"].isin(ARTIFACT_LABELS).sum())
        n_run_serie = int(group["run_series_flag"].eq(RUN_SERIE).sum()) if "run_series_flag" in group.columns else 0
        rows.append({
            "dense_region_id": int(dense_region_id),
            "index_debut": int(group.index.min()),
            "index_fin": int(group.index.max()),
            "debut_min": round(float(group["t_min"].min()), 2),
            "fin_min": round(float(group["t_min"].max()), 2),
            "nb_battements": n_points,
            "nb_artefacts": n_artifacts,
            "nb_points_run_serie": n_run_serie,
            "densite_artefact_pct": round((n_artifacts / n_points) * 100.0, 2) if n_points else 0.0,
            "segments_finaux": ", ".join(str(value) for value in segment_values) if segment_values else "-",
            "corrections": ", ".join(correction_values) if correction_values else "aucune",
        })
    return pd.DataFrame(rows)

def _apply_run_long_breaks(analysis: pd.DataFrame, params: RRCleaningParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = analysis.copy()
    rows: list[dict] = []
    ts = pd.to_datetime(out.get("timestamp"), errors="coerce") if "timestamp" in out.columns else pd.Series(pd.NaT, index=out.index)
    for start, end in _ranges(out["run_flag"].eq(RUN_LONG).to_numpy(dtype=bool)):
        out.loc[start:end, "run_flag"] = RUN_GAP_ARTEFACT
        out.loc[start:end, "correction_flag_raw"] = CORRECTION_NOT_CLEANED
        left_ok = next((idx for idx in range(start - 1, -1, -1) if out.loc[idx, "label"] == LABEL_OK), None)
        right_ok = next((idx for idx in range(end + 1, len(out)) if out.loc[idx, "label"] == LABEL_OK), None)
        duration_ms = 0.0
        if left_ok is not None and right_ok is not None and pd.notna(ts.iloc[left_ok]) and pd.notna(ts.iloc[right_ok]):
            duration_ms = float((ts.iloc[right_ok] - ts.iloc[left_ok]).total_seconds() * 1000.0)
        warmup = params.chauffe_gap_court if duration_ms < params.seuil_gap_duree_ms else params.chauffe_gap_long
        if right_ok is not None:
            count = 0
            idx = right_ok
            while idx < len(out) and count < warmup:
                if out.loc[idx, "deco_flag"] == DECO_OK and out.loc[idx, "run_flag"] == RUN_OK:
                    out.loc[idx, "run_flag"] = RUN_POST_RECONNECT_ARTEFACT
                    count += 1
                idx += 1
        rows.append({"break_type": BREAK_TYPE_ARTEFACT, "index_debut": int(start), "index_fin": int(end), "longueur_battements": int(end - start + 1), "duree_ms": round(duration_ms, 3), "flag_source": RUN_GAP_ARTEFACT})
    return out, pd.DataFrame(rows)

def _build_final_segments(analysis: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    seg_id = 0
    step_ms = analysis["source_timeline_step_ms"].to_numpy(dtype=float)
    ts = pd.to_datetime(analysis.get("timestamp"), errors="coerce") if "timestamp" in analysis.columns else pd.Series(pd.NaT, index=analysis.index)
    barrier = analysis["deco_flag"].eq(DECO_GAP_LONG) | analysis["run_flag"].eq(RUN_GAP_ARTEFACT)
    idx = 0
    while idx < len(analysis):
        if barrier.iloc[idx]:
            idx += 1
            continue
        start = idx
        while idx + 1 < len(analysis) and not barrier.iloc[idx + 1]:
            idx += 1
        end = idx
        seg_id += 1
        if pd.notna(ts.iloc[start]) and pd.notna(ts.iloc[end]):
            duration_ms = float((ts.iloc[end] - ts.iloc[start]).total_seconds() * 1000.0)
        else:
            duration_ms = float(step_ms[start : end + 1].sum())
        rows.append({"segment_final_id": seg_id, "index_debut": int(start), "index_fin": int(end), "duree_ms": round(duration_ms, 3)})
        idx += 1
    return pd.DataFrame(rows)

def _is_gap_point(record: pd.Series) -> bool:
    return record["deco_flag"] in {DECO_GAP_COURT, DECO_GAP_LONG}

def _is_non_ok_quality(record: pd.Series) -> bool:
    return record["label"] in ARTIFACT_LABELS or record.get("run_series_flag", RUN_OK) == RUN_SERIE or record["run_flag"] in POST_RECONNECT_RUN_FLAGS

def _quality_segments(analysis: pd.DataFrame, final_segments: pd.DataFrame, initial_segments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    initial_status = initial_segments.set_index("segment_id")["statut_segment"].to_dict() if not initial_segments.empty else {}
    for row in final_segments.itertuples(index=False):
        segment = analysis.loc[int(row.index_debut) : int(row.index_fin)]
        usable = segment.loc[~segment.apply(lambda rec: _is_gap_point(rec) or rec["run_flag"] == RUN_GAP_ARTEFACT, axis=1)]
        n_total = int(len(usable))
        n_art = int(usable.apply(_is_non_ok_quality, axis=1).sum())
        rate = (n_art / n_total) if n_total else np.nan
        initial_mode = int(segment["initial_segment_id"].mode().iloc[0]) if not segment.empty and segment["initial_segment_id"].notna().any() else 0
        rows.append({
            "segment_final_id": int(row.segment_final_id),
            "index_debut": int(row.index_debut),
            "index_fin": int(row.index_fin),
            "duree_ms": float(row.duree_ms),
            "n_total_seg": n_total,
            "n_artefacts": n_art,
            "n_battements_nets": max(n_total - n_art, 0),
            "taux_artefact": None if np.isnan(rate) else round(float(rate), 4),
            "qualite_segment": _quality(0.0 if np.isnan(rate) else float(rate)),
            "statut_segment_initial": initial_status.get(initial_mode),
        })
    return pd.DataFrame(rows)

def _global_quality(analysis: pd.DataFrame) -> tuple[float, str]:
    if analysis.empty:
        return 0.0, QUALITY_OK
    series_mask = analysis["run_series_flag"].eq(RUN_SERIE) if "run_series_flag" in analysis.columns else pd.Series(False, index=analysis.index)
    non_ok = analysis["label"].ne(LABEL_OK) | series_mask | analysis["run_flag"].isin(POST_RECONNECT_RUN_FLAGS)
    rate = float(non_ok.sum()) / float(len(analysis))
    return rate, _quality(rate)

def _choose_anchors(analysis: pd.DataFrame, start: int, end: int, y: int, tolerance: int) -> tuple[str, list[int]]:
    threshold = y - tolerance
    final_segment_id = int(analysis.loc[start, "segment_final_id"]) if pd.notna(analysis.loc[start, "segment_final_id"]) else 0
    same_segment = analysis["segment_final_id"].eq(final_segment_id)
    viable = same_segment & analysis["run_flag"].eq(RUN_OK) & analysis["label"].eq(LABEL_OK) & analysis["rr_interval_ms"].notna() & analysis["rr_interval_ms"].gt(0)
    before = analysis.index[viable & analysis.index.to_series().lt(start)].tolist()[-y:]
    after = analysis.index[viable & analysis.index.to_series().gt(end)].tolist()[:y]
    if len(before) >= threshold and len(after) >= threshold:
        return CORRECTION_INTERPOLATION_PCHIP, before + after
    if before and after:
        return CORRECTION_INTERPOLATION_LINEAIRE, [before[-1], after[0]]
    return CORRECTION_INTERPOLATION_LINEAIRE, []

def _interpolate_values(analysis: pd.DataFrame, start: int, end: int, n_output: int, method: str, anchors: list[int]) -> np.ndarray:
    med = analysis.loc[start:end, "med_locale_ms"].dropna().astype("float64")
    fallback = float(med.median()) if not med.empty else float(analysis.loc[start:end, "rr_interval_ms"].replace(0, np.nan).dropna().median())
    if np.isnan(fallback) or fallback <= 0:
        fallback = 1000.0
    positions = np.linspace(float(start), float(end), num=max(n_output, 1))
    if method == CORRECTION_INTERPOLATION_PCHIP and len(anchors) >= 4:
        x = np.array(anchors, dtype=float)
        y = analysis.loc[anchors, "rr_interval_ms"].astype("float64").to_numpy()
        try:
            values = PchipInterpolator(x, y)(positions)
            return np.where(values > 0, values, fallback).astype(float)
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            pass
    if anchors:
        left = anchors[0]
        right = anchors[-1]
        if left != right:
            values = np.interp(positions, [float(left), float(right)], [float(analysis.loc[left, "rr_interval_ms"]), float(analysis.loc[right, "rr_interval_ms"])] )
            return np.where(values > 0, values, fallback).astype(float)
    return np.full(max(n_output, 1), fallback, dtype=float)

def _timestamp_sequence(analysis: pd.DataFrame, start: int, end: int, n_output: int) -> tuple[list[pd.Timestamp | pd.NaT], list[float]]:
    ts = pd.to_datetime(analysis.get("timestamp"), errors="coerce") if "timestamp" in analysis.columns else pd.Series(pd.NaT, index=analysis.index)
    offsets = analysis["t_offset_ms"].astype("float64") if "t_offset_ms" in analysis.columns else pd.Series(np.nan, index=analysis.index)
    if n_output <= (end - start + 1):
        out_ts = [ts.iloc[idx] for idx in range(start, min(end + 1, start + n_output))]
        out_off = [float(offsets.iloc[idx]) if pd.notna(offsets.iloc[idx]) else np.nan for idx in range(start, min(end + 1, start + n_output))]
        while len(out_ts) < n_output:
            out_ts.append(out_ts[-1] if out_ts else pd.NaT)
            out_off.append(out_off[-1] if out_off else np.nan)
        return out_ts, out_off
    left_ts = ts.iloc[start]
    right_ts = ts.iloc[end]
    left_off = offsets.iloc[start]
    right_off = offsets.iloc[end]
    if pd.notna(left_ts) and pd.notna(right_ts) and left_ts != right_ts:
        fractions = np.linspace(0.0, 1.0, num=n_output)
        return [left_ts + (right_ts - left_ts) * frac for frac in fractions], [float(left_off + (right_off - left_off) * frac) if pd.notna(left_off) and pd.notna(right_off) else np.nan for frac in fractions]
    return [left_ts for _ in range(n_output)], [float(left_off) if pd.notna(left_off) else np.nan for _ in range(n_output)]

def _exploitability(run_flag: str, run_series_flag: str, correction_flag: str) -> tuple[bool, bool, bool, bool]:
    if correction_flag == CORRECTION_NOT_CLEANED:
        return False, False, False, False
    if run_flag == RUN_OK:
        if run_series_flag == RUN_SERIE:
            return True, True, False, False
        return True, True, True, False
    if run_flag in {RUN_ARTEFACT_UNIQUE, RUN_COURT}:
        return True, True, False, False
    if run_flag == RUN_MOYEN:
        return True, True, False, True
    if run_flag == RUN_POST_RECONNECT_DECO:
        return False, False, False, False
    if run_flag == RUN_POST_RECONNECT_ARTEFACT:
        return True, False, False, False
    if run_flag == RUN_GAP_DECO:
        return True, True, False, False
    return False, False, False, False

def _aggregate_label_2bis(analysis: pd.DataFrame, start: int, end: int) -> str:
    if "label_2bis" not in analysis.columns:
        return LABEL_2BIS_NONE
    values = analysis.loc[start:end, "label_2bis"].fillna(LABEL_2BIS_NONE).astype(str).tolist()
    if LABEL_2BIS_A in values:
        return LABEL_2BIS_A
    if LABEL_2BIS_B in values:
        return LABEL_2BIS_B
    return LABEL_2BIS_NONE

def _append_clean_rows(clean_rows: list[dict], values: list[float | None], timestamps: list[pd.Timestamp | pd.NaT], offsets: list[float], source_reference: str, label: str, label_2bis: str, run_flag: str, run_series_flag: str, deco_flag: str, correction_flag: str, segment_final_id: int, segment_status: str, source_index_start: int, source_index_end: int) -> None:
    for pos, value in enumerate(values):
        fc_ok, hrr_ok, rmssd_ok, review = _exploitability(run_flag, run_series_flag, correction_flag)
        clean_rows.append({
            "cleaned_index": len(clean_rows),
            "timestamp": timestamps[min(pos, len(timestamps) - 1)] if timestamps else pd.NaT,
            "t_offset_ms": offsets[min(pos, len(offsets) - 1)] if offsets else np.nan,
            "rr_interval_ms": value,
            "display_rr_ms": value if value is not None else np.nan,
            "source_reference": source_reference,
            "source_index_start": source_index_start,
            "source_index_end": source_index_end,
            "label": label,
            "label_2bis": label_2bis,
            "run_flag": run_flag,
            "run_series_flag": run_series_flag,
            "deco_flag": deco_flag,
            "correction_flag": correction_flag,
            "segment_final_id": segment_final_id,
            "segment_status": segment_status,
            "fc_ok": fc_ok and value is not None,
            "hrr_ok": hrr_ok and value is not None,
            "rmssd_ok": rmssd_ok and value is not None,
            "review_recommended": review,
        })

def _build_cleaned_frame(analysis: pd.DataFrame, quality_segments: pd.DataFrame, params: RRCleaningParams, progress_callback: Callable[[float], None] | None = None, progress_start: float = 0.0, progress_end: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_rows: list[dict] = []
    correction_raw = analysis["correction_flag_raw"].astype(str).tolist()
    quality_by_segment = quality_segments.set_index("segment_final_id")["qualite_segment"].to_dict() if not quality_segments.empty else {}
    ts = pd.to_datetime(analysis.get("timestamp"), errors="coerce") if "timestamp" in analysis.columns else pd.Series(pd.NaT, index=analysis.index)
    offsets = analysis["t_offset_ms"].astype("float64") if "t_offset_ms" in analysis.columns else pd.Series(np.nan, index=analysis.index)
    rr = analysis["rr_interval_ms"].astype("float64").to_numpy()
    idx = 0
    total_analysis = len(analysis)
    chunk = _progress_chunk(total_analysis)
    while idx < len(analysis):
        current_step = min(idx + 1, total_analysis)
        if total_analysis and (current_step % chunk == 0 or current_step == total_analysis):
            _emit_stage_progress(progress_callback, progress_start, progress_end, current_step, total_analysis)
        run_flag = str(analysis.loc[idx, "run_flag"])
        run_series_flag = str(analysis.loc[idx, "run_series_flag"]) if "run_series_flag" in analysis.columns else RUN_OK
        label = str(analysis.loc[idx, "label"])
        deco_flag = str(analysis.loc[idx, "deco_flag"])
        segment_status = str(analysis.loc[idx, "initial_segment_status"])
        segment_final_id = int(analysis.loc[idx, "segment_final_id"]) if pd.notna(analysis.loc[idx, "segment_final_id"]) else 0
        if segment_status == "exclu":
            correction_raw[idx] = CORRECTION_NOT_CLEANED
            _append_clean_rows(clean_rows, [None], [ts.iloc[idx]], [float(offsets.iloc[idx]) if pd.notna(offsets.iloc[idx]) else np.nan], str(idx + 1), label, _aggregate_label_2bis(analysis, idx, idx), run_flag, run_series_flag, deco_flag, CORRECTION_NOT_CLEANED, segment_final_id, segment_status, idx, idx)
            idx += 1
            continue
        if run_flag == RUN_GAP_DECO and deco_flag == DECO_GAP_COURT:
            start = idx
            while idx + 1 < len(analysis) and analysis.loc[idx + 1, "run_flag"] == RUN_GAP_DECO and analysis.loc[idx + 1, "deco_flag"] == DECO_GAP_COURT:
                idx += 1
            end = idx
            length = end - start + 1
            method, anchors = _choose_anchors(analysis, start, end, params.y_court if length <= params.seuil_run_court_max else params.y_moyen, params.tolerance_court if length <= params.seuil_run_court_max else params.tolerance_moyen)
            values = _interpolate_values(analysis, start, end, length, method, anchors).tolist()
            seq_ts, seq_offsets = _timestamp_sequence(analysis, start, end, length)
            for raw_idx in range(start, end + 1):
                correction_raw[raw_idx] = method
            _append_clean_rows(clean_rows, values, seq_ts, seq_offsets, f"{start + 1}-{end + 1}", LABEL_GAP_DECO, _aggregate_label_2bis(analysis, start, end), RUN_GAP_DECO, RUN_OK, DECO_GAP_COURT, method, segment_final_id, segment_status, start, end)
            idx += 1
            continue
        if run_flag in {RUN_GAP_DECO, RUN_GAP_ARTEFACT}:
            correction_raw[idx] = CORRECTION_NOT_CLEANED
            _append_clean_rows(clean_rows, [None], [ts.iloc[idx]], [float(offsets.iloc[idx]) if pd.notna(offsets.iloc[idx]) else np.nan], str(idx + 1), label, _aggregate_label_2bis(analysis, idx, idx), run_flag, run_series_flag, deco_flag, CORRECTION_NOT_CLEANED, segment_final_id, segment_status, idx, idx)
            idx += 1
            continue
        if run_flag in EXCLUDED_POST_RECONNECT_RUN_FLAGS:
            correction_raw[idx] = CORRECTION_NOT_CLEANED
            _append_clean_rows(clean_rows, [None], [ts.iloc[idx]], [float(offsets.iloc[idx]) if pd.notna(offsets.iloc[idx]) else np.nan], str(idx + 1), label, _aggregate_label_2bis(analysis, idx, idx), run_flag, run_series_flag, deco_flag, CORRECTION_NOT_CLEANED, segment_final_id, segment_status, idx, idx)
            idx += 1
            continue
        if run_flag in {RUN_OK, RUN_SERIE, RUN_POST_RECONNECT_ARTEFACT}:
            correction_raw[idx] = CORRECTION_OK
            value = float(rr[idx]) if pd.notna(rr[idx]) and rr[idx] > 0 else None
            _append_clean_rows(clean_rows, [value], [ts.iloc[idx]], [float(offsets.iloc[idx]) if pd.notna(offsets.iloc[idx]) else np.nan], str(idx + 1), label, _aggregate_label_2bis(analysis, idx, idx), run_flag, run_series_flag, deco_flag, CORRECTION_OK, segment_final_id, segment_status, idx, idx)
            idx += 1
            continue
        if run_flag == RUN_ARTEFACT_UNIQUE:
            if label == LABEL_MANQUE:
                value = float(rr[idx]) / 2.0 if pd.notna(rr[idx]) else None
                midpoint_ts = ts.iloc[idx] - pd.to_timedelta(value, unit="ms") if value is not None and pd.notna(ts.iloc[idx]) else ts.iloc[idx]
                base_offset = float(offsets.iloc[idx]) if pd.notna(offsets.iloc[idx]) else np.nan
                correction_raw[idx] = CORRECTION_DIVISION
                _append_clean_rows(clean_rows, [value, value], [midpoint_ts, ts.iloc[idx]], [base_offset - value if value is not None and not np.isnan(base_offset) else np.nan, base_offset], f"{idx + 1}a/{idx + 1}b", label, _aggregate_label_2bis(analysis, idx, idx), run_flag, run_series_flag, deco_flag, CORRECTION_DIVISION, segment_final_id, segment_status, idx, idx)
            elif label == LABEL_FAUX_BATTEMENT and idx + 1 < len(analysis):
                correction_raw[idx] = CORRECTION_FUSION
                correction_raw[idx + 1] = CORRECTION_NOT_CLEANED
                fused = float(rr[idx]) + float(rr[idx + 1]) if pd.notna(rr[idx]) and pd.notna(rr[idx + 1]) else None
                target_ts = ts.iloc[idx + 1] if idx + 1 < len(ts) else ts.iloc[idx]
                target_off = float(offsets.iloc[idx + 1]) if idx + 1 < len(offsets) and pd.notna(offsets.iloc[idx + 1]) else (float(offsets.iloc[idx]) if pd.notna(offsets.iloc[idx]) else np.nan)
                _append_clean_rows(clean_rows, [fused], [target_ts], [target_off], f"{idx + 1}+{idx + 2}", label, _aggregate_label_2bis(analysis, idx, min(idx + 1, len(analysis) - 1)), run_flag, run_series_flag, deco_flag, CORRECTION_FUSION, segment_final_id, segment_status, idx, min(idx + 1, len(analysis) - 1))
            else:
                method, anchors = _choose_anchors(analysis, idx, idx, params.y_court, params.tolerance_court)
                value = float(_interpolate_values(analysis, idx, idx, 1, method, anchors)[0])
                correction_raw[idx] = method
                _append_clean_rows(clean_rows, [value], [ts.iloc[idx]], [float(offsets.iloc[idx]) if pd.notna(offsets.iloc[idx]) else np.nan], str(idx + 1), label, _aggregate_label_2bis(analysis, idx, idx), run_flag, run_series_flag, deco_flag, method, segment_final_id, segment_status, idx, idx)
            idx += 1
            continue
        if run_flag in {RUN_COURT, RUN_MOYEN}:
            current_flag = run_flag
            start = idx
            while idx + 1 < len(analysis) and analysis.loc[idx + 1, "run_flag"] == current_flag:
                idx += 1
            end = idx
            block_labels = analysis.loc[start:end, "label"]
            n_output = max((end - start + 1) + int(block_labels.eq(LABEL_MANQUE).sum()) - int(block_labels.eq(LABEL_FAUX_BATTEMENT).sum()), 1)
            method, anchors = _choose_anchors(analysis, start, end, params.y_court if current_flag == RUN_COURT else params.y_moyen, params.tolerance_court if current_flag == RUN_COURT else params.tolerance_moyen)
            values = _interpolate_values(analysis, start, end, n_output, method, anchors).tolist()
            seq_ts, seq_offsets = _timestamp_sequence(analysis, start, end, n_output)
            for raw_idx in range(start, end + 1):
                correction_raw[raw_idx] = method
            block_series_flag = RUN_SERIE if analysis.loc[start:end, "run_series_flag"].eq(RUN_SERIE).any() else RUN_OK
            _append_clean_rows(clean_rows, values, seq_ts, seq_offsets, f"{start + 1}-{end + 1}", str(analysis.loc[start, "label"]), _aggregate_label_2bis(analysis, start, end), current_flag, block_series_flag, str(analysis.loc[start, "deco_flag"]), method, segment_final_id, segment_status, start, end)
            idx += 1
            continue
        correction_raw[idx] = CORRECTION_OK
        value = float(rr[idx]) if pd.notna(rr[idx]) and rr[idx] > 0 else None
        _append_clean_rows(clean_rows, [value], [ts.iloc[idx]], [float(offsets.iloc[idx]) if pd.notna(offsets.iloc[idx]) else np.nan], str(idx + 1), label, _aggregate_label_2bis(analysis, idx, idx), run_flag, run_series_flag, deco_flag, CORRECTION_OK, segment_final_id, segment_status, idx, idx)
        idx += 1
    cleaned = pd.DataFrame(clean_rows)
    if not cleaned.empty:
        cleaned["quality_segment_label"] = cleaned["segment_final_id"].map(lambda value: quality_segments.set_index("segment_final_id")["qualite_segment"].to_dict().get(int(value), QUALITY_OK) if int(value) > 0 else QUALITY_EXCLUSION)
        cleaned["timeline_step_ms"] = cleaned["rr_interval_ms"].fillna(analysis["source_timeline_step_ms"].median()).astype("float64")
    out = analysis.copy()
    out["correction_flag_raw"] = correction_raw
    _emit_progress(progress_callback, progress_end)
    return cleaned, out

def analyze_rr_artifacts(frame: pd.DataFrame, params: RRCleaningParams | None = None, bridge_decisions: dict[str, bool] | None = None, labels_override: list[str] | None = None, progress_callback: Callable[[float], None] | None = None) -> RRCleaningResult:
    del bridge_decisions
    params = params or RRCleaningParams()
    emit_progress = _wrap_progress_callback(progress_callback)
    if frame.empty:
        emit_progress(1.0)
        return _empty_result(frame)
    emit_progress(0.02)
    analysis = frame.copy().reset_index(drop=True)
    analysis["rr_interval_ms"] = analysis["rr_interval_ms"].astype("float64")
    analysis["source_pipeline_flag"] = _source_flags(analysis)
    analysis["source_timeline_step_ms"] = _source_step_ms(analysis)
    analysis["deco_flag"], deco_breaks = _build_deco_flags(analysis, params)
    emit_progress(0.10)
    analysis["label_initial"] = LABEL_A_VERIFIER
    analysis.loc[analysis["deco_flag"].ne(DECO_OK), "label_initial"] = LABEL_GAP_DECO
    analysis.loc[analysis["deco_flag"].eq(DECO_OK) & ((analysis["rr_interval_ms"] < params.rr_min_ms) | (analysis["rr_interval_ms"] > params.rr_max_ms)), "label_initial"] = LABEL_ARTEFACT_ABSOLU
    analysis["label"] = analysis["label_initial"]
    analysis["label_2bis"] = LABEL_2BIS_NONE
    analysis["correction_flag_raw"] = CORRECTION_OK

    initial_segments = _build_initial_segments(analysis, params)
    emit_progress(0.18)
    analysis["initial_segment_id"] = 0
    for row in initial_segments.itertuples(index=False):
        analysis.loc[int(row.index_debut) : int(row.index_fin), "initial_segment_id"] = int(row.segment_id)

    analysis = _classify_labels(analysis, params, progress_callback=emit_progress, progress_start=0.18, progress_end=0.45)
    if labels_override is not None:
        labels_override_list = list(labels_override)
        if len(labels_override_list) != len(analysis):
            raise ValueError("labels_override must have the same length as the input RR frame")
        analysis["label"] = pd.Series(labels_override_list, index=analysis.index, dtype="object")
    initial_segments = _apply_segment_inclusion(analysis, initial_segments)
    emit_progress(0.55)
    initial_status = initial_segments.set_index("segment_id")["statut_segment"].to_dict() if not initial_segments.empty else {}
    analysis["initial_segment_status"] = analysis["initial_segment_id"].map(lambda value: initial_status.get(int(value), "exclu") if int(value) > 0 else "exclu")
    analysis.loc[analysis["initial_segment_status"].eq("exclu"), "correction_flag_raw"] = CORRECTION_NOT_CLEANED

    analysis = _assign_run_flags(analysis, initial_segments, params, progress_callback=emit_progress, progress_start=0.55, progress_end=0.68)
    analysis = _mark_dense_regions(analysis)
    analysis, artefact_breaks = _apply_run_long_breaks(analysis, params)
    emit_progress(0.76)

    final_segments = _build_final_segments(analysis)
    emit_progress(0.84)
    analysis["segment_final_id"] = 0
    for row in final_segments.itertuples(index=False):
        analysis.loc[int(row.index_debut) : int(row.index_fin), "segment_final_id"] = int(row.segment_final_id)

    quality_segments = _quality_segments(analysis, final_segments, initial_segments)
    emit_progress(0.90)
    cleaned_frame, analysis = _build_cleaned_frame(analysis, quality_segments, params, progress_callback=emit_progress, progress_start=0.90, progress_end=0.98)
    dense_map = analysis["dense_region_id"].to_dict() if "dense_region_id" in analysis.columns else {}
    if not cleaned_frame.empty:
        cleaned_frame["dense_region_id"] = cleaned_frame["source_index_start"].map(lambda idx: int(dense_map.get(int(idx), 0)))
        cleaned_frame["is_dense_zone"] = cleaned_frame["dense_region_id"].gt(0)

    dense_regions_frame = _build_dense_regions_frame(analysis, cleaned_frame)

    breaks_frame = pd.concat([deco_breaks, artefact_breaks], ignore_index=True) if not deco_breaks.empty or not artefact_breaks.empty else pd.DataFrame(columns=["break_type", "index_debut", "index_fin", "longueur_battements", "duree_ms", "flag_source"])
    segments_frame = initial_segments.rename(columns={"segment_id": "segment_initial_id"}).copy() if not initial_segments.empty else pd.DataFrame()
    label_counts = _counts(analysis["label"], LABEL_ORDER)
    run_flag_counts = _counts(analysis["run_flag"], RUN_FLAG_ORDER)
    run_series_flag_counts = _counts(analysis["run_series_flag"], RUN_SERIES_FLAG_ORDER) if "run_series_flag" in analysis.columns else {}
    deco_flag_counts = _counts(analysis["deco_flag"], DECO_FLAG_ORDER)
    correction_counts = _counts(cleaned_frame["correction_flag"], CORRECTION_FLAG_ORDER) if not cleaned_frame.empty else {}

    n_cassures_deco = int(len(deco_breaks.loc[deco_breaks["flag_source"].eq(DECO_GAP_LONG)])) if not deco_breaks.empty else 0
    n_cassures_artefact = int(len(artefact_breaks)) if not artefact_breaks.empty else 0
    n_cassures_total = n_cassures_deco + n_cassures_artefact
    dense_points_count = int(analysis["run_series_flag"].eq(RUN_SERIE).sum()) if "run_series_flag" in analysis.columns else 0
    n_zones_denses = int(analysis["dense_region_id"].max()) if "dense_region_id" in analysis.columns and len(analysis) else 0
    n_segments_actifs = int(initial_segments["statut_segment"].eq("actif").sum()) if not initial_segments.empty else 0
    n_segments_exclus = int(initial_segments["statut_segment"].eq("exclu").sum()) if not initial_segments.empty else 0
    eligible = analysis.loc[analysis["initial_segment_status"].eq("actif") & ~analysis["run_flag"].isin([RUN_GAP_DECO, RUN_GAP_ARTEFACT])]
    corrected = cleaned_frame.loc[cleaned_frame["correction_flag"].isin([CORRECTION_DIVISION, CORRECTION_FUSION, CORRECTION_INTERPOLATION_PCHIP, CORRECTION_INTERPOLATION_LINEAIRE])] if not cleaned_frame.empty else pd.DataFrame()
    correction_rate = float(len(corrected)) / float(len(eligible)) if len(eligible) else 0.0
    global_non_ok_rate, global_quality_label = _global_quality(analysis)
    ok_rr_total = int(cleaned_frame["fc_ok"].sum()) if not cleaned_frame.empty else 0
    non_viable_rr_total = int((~cleaned_frame[["fc_ok", "hrr_ok", "rmssd_ok"]].any(axis=1)).sum()) if not cleaned_frame.empty else 0

    emit_progress(1.0)
    return RRCleaningResult(
        analysis_frame=analysis,
        cleaned_frame=cleaned_frame,
        segments_frame=segments_frame,
        quality_segments_frame=quality_segments,
        breaks_frame=breaks_frame,
        dense_regions_frame=dense_regions_frame,
        label_counts=label_counts,
        run_flag_counts=run_flag_counts,
        run_series_flag_counts=run_series_flag_counts,
        deco_flag_counts=deco_flag_counts,
        correction_counts=correction_counts,
        total_points=int(len(analysis)),
        cleaned_points=int(cleaned_frame["rr_interval_ms"].notna().sum()) if not cleaned_frame.empty else 0,
        ok_rr_total=ok_rr_total,
        non_viable_rr_total=non_viable_rr_total,
        n_cassures_total=n_cassures_total,
        n_cassures_deco=n_cassures_deco,
        n_cassures_artefact=n_cassures_artefact,
        n_zones_denses=n_zones_denses,
        dense_points_count=dense_points_count,
        n_segments_actifs=n_segments_actifs,
        n_segments_exclus=n_segments_exclus,
        correction_rate_outside_breaks=correction_rate,
        global_non_ok_rate=global_non_ok_rate,
        global_quality_label=global_quality_label,
    )


