from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd

from polar_app.models import ProcessedSession
from polar_app.repository import ProcessedSessionRepository
from polar_app.rr_pipeline import RRCleaningParams, RRCleaningResult, analyze_rr_artifacts

FC_CLEAN_WINDOW_BEATS = 5
FC_CLEAN_MIN_VIABLE_POINTS = 2


def _session_start(session: ProcessedSession) -> datetime:
    if session.fc_start_ts:
        return datetime.fromisoformat(session.fc_start_ts)
    return datetime.fromisoformat(f"{session.date}T{session.heure_debut}")


def build_rr_clean_export(session: ProcessedSession, result: RRCleaningResult) -> pd.DataFrame:
    cleaned = result.cleaned_frame.copy().reset_index(drop=True)
    if cleaned.empty:
        return cleaned

    start_dt = _session_start(session)
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
    cleaned["t_offset_ms"] = cleaned["t_offset_ms"].astype("float64")
    cleaned["timeline_step_ms"] = cleaned["timeline_step_ms"].astype("float64")

    fallback_offsets = cleaned["timeline_step_ms"].cumsum()
    missing_mask = cleaned["timestamp"].isna()
    if missing_mask.any():
        cleaned.loc[missing_mask, "t_offset_ms"] = fallback_offsets.loc[missing_mask]
        cleaned.loc[missing_mask, "timestamp"] = start_dt + pd.to_timedelta(cleaned.loc[missing_mask, "t_offset_ms"], unit="ms")

    cleaned["t_offset_clean_ms"] = ((cleaned["timestamp"] - start_dt).dt.total_seconds() * 1000.0).round()
    cleaned["session_id"] = session.session_id
    cleaned["session_start_ts"] = start_dt.isoformat()
    cleaned["display_rr_ms"] = cleaned["rr_interval_ms"].astype("float64")

    ordered_columns = [
        "session_id",
        "cleaned_index",
        "timestamp",
        "t_offset_clean_ms",
        "rr_interval_ms",
        "display_rr_ms",
        "label",
        "run_flag",
        "deco_flag",
        "correction_flag",
        "segment_final_id",
        "segment_status",
        "quality_segment_label",
        "fc_ok",
        "hrr_ok",
        "rmssd_ok",
        "review_recommended",
        "dense_region_id",
        "source_reference",
        "source_index_start",
        "source_index_end",
        "timeline_step_ms",
        "session_start_ts",
    ]
    return cleaned[[column for column in ordered_columns if column in cleaned.columns]].copy()


def build_fc_clean_export(rr_clean_frame: pd.DataFrame, window_beats: int = FC_CLEAN_WINDOW_BEATS, min_viable_points: int = FC_CLEAN_MIN_VIABLE_POINTS) -> pd.DataFrame:
    if rr_clean_frame.empty:
        return pd.DataFrame(columns=["cleaned_index", "timestamp", "t_offset_clean_ms", "bpm_clean", "window_viable_points", "window_size_beats"])

    half_window = window_beats // 2
    fc_rows: list[dict] = []
    for idx in range(len(rr_clean_frame)):
        start = max(0, idx - half_window)
        end = min(len(rr_clean_frame), idx + half_window + 1)
        window = rr_clean_frame.iloc[start:end]
        viable = window.loc[window["fc_ok"].eq(True) & window["rr_interval_ms"].notna(), "rr_interval_ms"].astype("float64")
        bpm_clean = np.nan
        if len(viable) >= min_viable_points:
            mean_rr = float(viable.mean())
            if mean_rr > 0:
                bpm_clean = 60000.0 / mean_rr
        fc_rows.append(
            {
                "cleaned_index": int(rr_clean_frame.iloc[idx]["cleaned_index"]),
                "timestamp": rr_clean_frame.iloc[idx]["timestamp"],
                "t_offset_clean_ms": float(rr_clean_frame.iloc[idx]["t_offset_clean_ms"]),
                "bpm_clean": bpm_clean,
                "window_viable_points": int(len(viable)),
                "window_size_beats": int(window_beats),
                "center_fc_ok": bool(rr_clean_frame.iloc[idx].get("fc_ok", False)),
                "center_hrr_ok": bool(rr_clean_frame.iloc[idx].get("hrr_ok", False)),
                "center_rmssd_ok": bool(rr_clean_frame.iloc[idx].get("rmssd_ok", False)),
            }
        )
    return pd.DataFrame(fc_rows)


def export_clean_result(repository: ProcessedSessionRepository, session: ProcessedSession, result: RRCleaningResult, params: RRCleaningParams) -> dict[str, str]:
    rr_clean_frame = build_rr_clean_export(session, result)
    fc_clean_frame = build_fc_clean_export(rr_clean_frame)
    clean_meta = {
        "session_id": session.session_id,
        "source_rr_filepath": session.rr_filepath,
        "source_hr_filepath": session.hr_filepath,
        "export_timestamp": datetime.now().isoformat(),
        "fc_clean_window_beats": FC_CLEAN_WINDOW_BEATS,
        "fc_clean_min_viable_points": FC_CLEAN_MIN_VIABLE_POINTS,
        "ok_rr_total": int(result.ok_rr_total),
        "non_viable_rr_total": int(result.non_viable_rr_total),
        "global_non_ok_rate": float(result.global_non_ok_rate),
        "global_quality_label": result.global_quality_label,
        "params": asdict(params),
    }
    return repository.save_clean_export(session.session_id, rr_clean_frame, fc_clean_frame, clean_meta)


def export_all_sessions_clean(output_dir: str, params: RRCleaningParams | None = None) -> dict[str, list[str]]:
    repository = ProcessedSessionRepository(output_dir)
    params = params or RRCleaningParams()
    exported: list[str] = []
    skipped: list[str] = []
    for session in repository.list_sessions():
        try:
            session_obj, rr_frame, _ = repository.load_session_data(session.session_id)
            result = analyze_rr_artifacts(rr_frame, params)
            export_clean_result(repository, session_obj, result, params)
            exported.append(session.session_id)
        except Exception:
            skipped.append(session.session_id)
    return {"exported": exported, "skipped": skipped}
