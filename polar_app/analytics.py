from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_axis(frame: pd.DataFrame, offset_column: str = "t_offset_ms") -> pd.DataFrame:
    """
    Retourne une copie avec des colonnes temps utiles pour l'affichage.
    """
    enriched = frame.copy()
    if offset_column not in enriched.columns:
        raise ValueError(f"Colonne temps introuvable : {offset_column}")

    enriched["t_s"] = enriched[offset_column] / 1000
    enriched["t_min"] = enriched["t_s"] / 60
    return enriched


def _safe_percent(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator) * 100, 1)


def _estimate_missing_points(frame: pd.DataFrame, value_column: str, threshold_ms: int = 3000) -> tuple[int, float]:
    if frame.empty or "t_offset_ms" not in frame.columns or value_column not in frame.columns:
        return 0, 0.0

    enriched = frame.sort_values("t_offset_ms").reset_index(drop=True).copy()
    valid_mask = enriched[value_column].fillna(0) > 0
    valid_frame = enriched.loc[valid_mask].copy()
    if len(valid_frame) < 2:
        return 0, 0.0

    diffs = valid_frame["t_offset_ms"].diff()
    regular_diffs = diffs[(diffs > 0) & (diffs <= threshold_ms)]
    typical_step = float(regular_diffs.median()) if not regular_diffs.empty else 1000.0
    if typical_step <= 0:
        typical_step = 1000.0

    gap_diffs = diffs[diffs > threshold_ms]
    if gap_diffs.empty:
        return 0, 0.0

    missing_points = int(sum(max(int(round(diff / typical_step)) - 1, 0) for diff in gap_diffs))
    total_expected = len(valid_frame) + missing_points
    return missing_points, _safe_percent(missing_points, total_expected)


def compute_rr_missing_stats(frame: pd.DataFrame) -> tuple[int, float]:
    if frame.empty or "rr_interval_ms" not in frame.columns:
        return 0, 0.0

    if "pipeline_flag" in frame.columns:
        missing_points = int(frame["pipeline_flag"].eq("GAP").sum())
    else:
        missing_points = int((frame["rr_interval_ms"].fillna(0) <= 0).sum())

    total_points = len(frame)
    return missing_points, _safe_percent(missing_points, total_points)


def compute_hr_missing_stats(frame: pd.DataFrame) -> tuple[int, float]:
    return _estimate_missing_points(frame, "bpm")


def build_rr_chart_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise le DataFrame RR pour un affichage graphique.
    Les gaps et valeurs a 0 sont exclus, et les portions sont groupees pour casser la courbe.
    """
    enriched = add_time_axis(frame)
    enriched["rr_interval_ms"] = enriched["rr_interval_ms"].where(enriched["rr_interval_ms"] > 0, np.nan)
    enriched["bpm_instantane"] = enriched["bpm_instantane"].where(enriched["bpm_instantane"] > 0, np.nan)

    break_mask = pd.Series(False, index=enriched.index)
    if "pipeline_flag" in enriched.columns:
        break_mask = break_mask | enriched["pipeline_flag"].eq("GAP")
    if "segment_id" in enriched.columns:
        break_mask = break_mask | enriched["segment_id"].ne(enriched["segment_id"].shift(1)).fillna(False)

    chart = enriched[["t_min", "rr_interval_ms", "bpm_instantane"]].copy()
    chart["line_group"] = break_mask.astype(int).cumsum()
    return chart


def build_hr_chart_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise le DataFrame FC pour un affichage graphique.
    Les valeurs nulles sont exclues et les longues deconnexions cassent la courbe.
    """
    enriched = add_time_axis(frame)
    enriched["bpm"] = enriched["bpm"].where(enriched["bpm"] > 0, np.nan)

    break_mask = pd.Series(False, index=enriched.index)
    if "segment_id" in enriched.columns:
        break_mask = break_mask | enriched["segment_id"].ne(enriched["segment_id"].shift(1)).fillna(False)
    if "t_offset_ms" in enriched.columns:
        break_mask = break_mask | enriched["t_offset_ms"].diff().gt(3000).fillna(False)

    chart = enriched[["t_min", "bpm"]].copy()
    chart["line_group"] = break_mask.astype(int).cumsum()
    return chart
