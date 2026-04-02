from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from polar_app.models import ProcessedSession


class ProcessedSessionRepository:
    FAMILY_COLORS = {
        "judo": ("#2d6a4f", "#1b4332"),
        "prepa": ("#b26a1b", "#8f4e0d"),
        "autres": ("#6b7280", "#4b5563"),
        None: ("#94a3b8", "#64748b"),
    }
    TEMPORAL_SEGMENT_LABELS = [
        "echauffement",
        "technique",
        "randori",
        "recuperation",
        "retour_calme",
        "autre",
    ]
    MIN_SEGMENT_DURATION_S = 10.0

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.processed_root = os.path.join(output_dir, "processed")
        self.clean_root = os.path.join(output_dir, "clean")

    def list_sessions(self, include_archived: bool = False) -> list[ProcessedSession]:
        sessions: list[ProcessedSession] = []

        if not os.path.isdir(self.processed_root):
            return sessions

        for session_id in sorted(os.listdir(self.processed_root)):
            meta_path = os.path.join(self.processed_root, session_id, "session_meta.json")
            if not os.path.isfile(meta_path):
                continue

            with open(meta_path, encoding="utf-8-sig") as file_obj:
                session = ProcessedSession.from_dict(json.load(file_obj))
            clean_paths = self.get_clean_paths(session.session_id)
            if os.path.isfile(clean_paths["rr_clean_absolute"]):
                session.rr_clean_filepath = clean_paths["rr_clean_relative"]
            if os.path.isfile(clean_paths["fc_clean_absolute"]):
                session.fc_clean_filepath = clean_paths["fc_clean_relative"]
            clean_meta_path = clean_paths["clean_meta_absolute"]
            if os.path.isfile(clean_meta_path):
                with open(clean_meta_path, encoding="utf-8-sig") as clean_file_obj:
                    clean_meta = json.load(clean_file_obj)
                session.clean_export_timestamp = clean_meta.get("export_timestamp")
            session.is_archived = bool(session.is_archived) if session.is_archived is not None else False
            if session.is_activity_annotated is None:
                session.is_activity_annotated = bool(session.activity_family and session.activity_label)
            else:
                session.is_activity_annotated = bool(session.is_activity_annotated)
            if session.is_temporally_annotated is None:
                session.is_temporally_annotated = bool(
                    self._is_judo_randoris_session(session) and session.fc_phase_segments
                )
            else:
                session.is_temporally_annotated = bool(session.is_temporally_annotated)
            if session.is_archived and not include_archived:
                continue
            sessions.append(session)

        return sessions

    def get_sessions(self, session_ids: list[str] | None = None, include_archived: bool = False) -> list[ProcessedSession]:
        sessions = self.list_sessions(include_archived=include_archived)
        if session_ids is None:
            return sessions

        sessions_map = {session.session_id: session for session in sessions}
        missing = [session_id for session_id in session_ids if session_id not in sessions_map]
        if missing:
            raise ValueError("Session(s) introuvable(s) : " + ", ".join(missing))

        return [sessions_map[session_id] for session_id in session_ids]

    def load_dataframe(self, filepath: str) -> pd.DataFrame:
        absolute_path = os.path.join(self.output_dir, filepath)
        extension = os.path.splitext(absolute_path)[1].lower()

        if extension == ".csv":
            return pd.read_csv(absolute_path)
        if extension == ".parquet":
            return pd.read_parquet(absolute_path)

        raise ValueError(f"Format de fichier non supporte : {absolute_path}")

    def load_session_data(self, session_id: str) -> tuple[ProcessedSession, pd.DataFrame, pd.DataFrame]:
        session = self.get_sessions([session_id], include_archived=True)[0]
        rr_frame = self.load_dataframe(session.rr_filepath)
        hr_frame = self.load_dataframe(session.hr_filepath)
        return session, rr_frame, hr_frame

    def get_clean_paths(self, session_id: str) -> dict[str, str]:
        clean_dir = os.path.join(self.clean_root, session_id)
        return {
            "clean_dir": clean_dir,
            "rr_clean_absolute": os.path.join(clean_dir, "rr_clean.parquet"),
            "fc_clean_absolute": os.path.join(clean_dir, "fc_clean.parquet"),
            "clean_meta_absolute": os.path.join(clean_dir, "clean_meta.json"),
            "rr_clean_relative": os.path.join("clean", session_id, "rr_clean.parquet"),
            "fc_clean_relative": os.path.join("clean", session_id, "fc_clean.parquet"),
            "clean_meta_relative": os.path.join("clean", session_id, "clean_meta.json"),
        }

    def has_clean_export(self, session_id: str) -> bool:
        paths = self.get_clean_paths(session_id)
        return os.path.isfile(paths["rr_clean_absolute"]) and os.path.isfile(paths["fc_clean_absolute"])

    def load_clean_data(self, session_id: str) -> tuple[ProcessedSession, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        session = self.get_sessions([session_id], include_archived=True)[0]
        paths = self.get_clean_paths(session_id)
        if not os.path.isfile(paths["rr_clean_absolute"]) or not os.path.isfile(paths["fc_clean_absolute"]):
            raise FileNotFoundError(f"Export clean introuvable pour la session {session_id}")
        rr_clean = self.load_dataframe(paths["rr_clean_relative"])
        fc_clean = self.load_dataframe(paths["fc_clean_relative"])
        clean_meta: dict[str, Any] = {}
        if os.path.isfile(paths["clean_meta_absolute"]):
            with open(paths["clean_meta_absolute"], encoding="utf-8-sig") as file_obj:
                clean_meta = json.load(file_obj)
        return session, rr_clean, fc_clean, clean_meta

    def save_clean_export(
        self,
        session_id: str,
        rr_clean_frame: pd.DataFrame,
        fc_clean_frame: pd.DataFrame,
        clean_meta: dict[str, Any],
    ) -> dict[str, str]:
        paths = self.get_clean_paths(session_id)
        os.makedirs(paths["clean_dir"], exist_ok=True)
        rr_clean_frame.to_parquet(paths["rr_clean_absolute"], index=False)
        fc_clean_frame.to_parquet(paths["fc_clean_absolute"], index=False)
        with open(paths["clean_meta_absolute"], "w", encoding="utf-8") as file_obj:
            json.dump(clean_meta, file_obj, ensure_ascii=False, indent=2)

        processed_meta_path = os.path.join(self.processed_root, session_id, "session_meta.json")
        if os.path.isfile(processed_meta_path):
            with open(processed_meta_path, encoding="utf-8-sig") as file_obj:
                processed_meta = json.load(file_obj)
            processed_meta["rr_clean_filepath"] = paths["rr_clean_relative"]
            processed_meta["fc_clean_filepath"] = paths["fc_clean_relative"]
            processed_meta["clean_export_timestamp"] = clean_meta.get("export_timestamp")
            with open(processed_meta_path, "w", encoding="utf-8") as file_obj:
                json.dump(processed_meta, file_obj, ensure_ascii=False, indent=2)
        return paths

    def update_annotation(self, session_id: str, annotation: str) -> None:
        self.update_activity_metadata(session_id, {"annotation": annotation})

    def update_activity_metadata(self, session_id: str, payload: dict[str, Any]) -> None:
        session = self.get_sessions([session_id], include_archived=True)[0]
        meta = self._read_session_meta(session_id)
        now = datetime.now().isoformat()

        annotation = self._normalize_string(payload.get("annotation"))
        activity_family = self._normalize_string(payload.get("activity_family"))
        activity_label = self._normalize_string(payload.get("activity_label"))
        activity_notes = self._normalize_string(payload.get("activity_notes"))
        session_rpe = self._normalize_optional_int(payload.get("session_rpe"))
        judo_session_type = self._normalize_string(payload.get("judo_session_type")) if activity_family == "judo" else None
        judo_phases = self._normalize_dict_list(payload.get("judo_phases")) if activity_family == "judo" and judo_session_type == "randoris" else None
        judo_randori_blocks = self._normalize_dict_list(payload.get("judo_randori_blocks")) if activity_family == "judo" and judo_session_type == "randoris" else None
        is_activity_annotated = bool(activity_family and activity_label)
        activity_annotated_at = meta.get("activity_annotated_at")
        if is_activity_annotated and not activity_annotated_at:
            activity_annotated_at = now
        if not is_activity_annotated:
            activity_annotated_at = None

        is_judo_randoris = activity_family == "judo" and judo_session_type == "randoris"

        meta["annotation"] = annotation or session.annotation or session.session_id
        meta["activity_family"] = activity_family
        meta["activity_label"] = activity_label
        meta["activity_notes"] = activity_notes
        meta["session_rpe"] = session_rpe
        meta["judo_session_type"] = judo_session_type
        meta["judo_phases"] = judo_phases
        meta["judo_randori_blocks"] = judo_randori_blocks
        meta["is_activity_annotated"] = is_activity_annotated
        meta["activity_annotated_at"] = activity_annotated_at
        meta["is_archived"] = bool(meta.get("is_archived", False))
        if not is_judo_randoris:
            meta["fc_phase_segments"] = None
            meta["is_temporally_annotated"] = False
            meta["temporally_annotated_at"] = None
        else:
            meta["is_temporally_annotated"] = bool(meta.get("is_temporally_annotated", False))
        meta["updated_at"] = now
        self._write_session_meta(session_id, meta)

    def archive_session(self, session_id: str) -> None:
        meta = self._read_session_meta(session_id)
        now = datetime.now().isoformat()
        meta["is_archived"] = True
        meta["archived_at"] = now
        meta["updated_at"] = now
        self._write_session_meta(session_id, meta)

    def restore_session(self, session_id: str) -> None:
        meta = self._read_session_meta(session_id)
        meta["is_archived"] = False
        meta["archived_at"] = None
        meta["updated_at"] = datetime.now().isoformat()
        self._write_session_meta(session_id, meta)

    def build_calendar_events(
        self,
        include_archived: bool = False,
        family_filter: str | None = None,
        annotated_only: bool = False,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for session in self.list_sessions(include_archived=include_archived):
            if family_filter and family_filter != "toutes" and session.activity_family != family_filter:
                continue
            if annotated_only and not session.is_activity_annotated:
                continue

            start_dt = self.get_session_start(session)
            if start_dt is None:
                continue
            end_dt = self.get_session_end(session) or (start_dt + timedelta(hours=1))
            background_color, border_color = self.FAMILY_COLORS.get(session.activity_family, self.FAMILY_COLORS[None])
            if session.is_archived:
                background_color = "#cbd5e1"
                border_color = "#94a3b8"
            if not session.is_activity_annotated:
                border_color = "#ef4444"

            events.append(
                {
                    "id": session.session_id,
                    "title": session.annotation or session.session_id,
                    "start": start_dt.isoformat(),
                    "end": end_dt.isoformat(),
                    "backgroundColor": background_color,
                    "borderColor": border_color,
                    "textColor": "#102a1d",
                    "extendedProps": {
                        "annotated": bool(session.is_activity_annotated),
                        "archived": bool(session.is_archived),
                        "activity_family": session.activity_family,
                        "activity_label": session.activity_label,
                        "session_id": session.session_id,
                    },
                }
            )
        return events

    def generate_default_fc_phase_segments(
        self,
        session_id: str,
        threshold_bpm: float = 160.0,
        default_recovery_min: float = 2.0,
    ) -> list[dict[str, Any]]:
        session, _, hr_frame = self.load_session_data(session_id)
        if not self._is_judo_randoris_session(session):
            raise ValueError("Les annotations temporelles automatiques sont reservees aux seances judo randoris.")

        if hr_frame.empty or "bpm" not in hr_frame.columns or "t_offset_ms" not in hr_frame.columns:
            raise ValueError("Aucune donnee FC exploitable pour generer les segments temporels.")

        phase_entries = self._build_judo_phase_entries(session)
        if not phase_entries:
            raise ValueError("Impossible d'auto-generer : aucune phase judo pre-enregistree n'a ete trouvee pour cette seance.")

        hr_sorted = hr_frame.sort_values("t_offset_ms").reset_index(drop=True).copy()
        hr_sorted["t_offset_s"] = hr_sorted["t_offset_ms"].astype("float64") / 1000.0
        total_duration_s = self._session_total_duration_seconds(session, hr_sorted)

        if not any(phase["label"] == "randori" for phase in phase_entries):
            raise ValueError("Impossible d'auto-generer : aucune phase randoris n'a ete pre-enregistree pour cette seance.")

        def coerce_positive_int(value: Any) -> int | None:
            if value in (None, "", "NA"):
                return None
            try:
                candidate = int(float(value))
            except (TypeError, ValueError):
                return None
            return candidate if candidate > 0 else None

        def coerce_positive_seconds(value_min: Any) -> float | None:
            if value_min in (None, "", "NA"):
                return None
            try:
                candidate = float(value_min) * 60.0
            except (TypeError, ValueError):
                return None
            return candidate if candidate > 0 else None

        segments: list[dict[str, Any]] = []
        pending_non_randori: list[dict[str, Any]] = []
        current_cursor_s = 0.0

        for phase_entry in phase_entries:
            if phase_entry["label"] != "randori":
                pending_non_randori.append(phase_entry)
                continue

            detected_randori_start_s = self._detect_randori_start(hr_sorted, current_cursor_s, threshold_bpm)
            min_gap_s = float(len(pending_non_randori)) * self.MIN_SEGMENT_DURATION_S
            randori_start_s = min(
                max(detected_randori_start_s, current_cursor_s + min_gap_s),
                total_duration_s,
            )

            if pending_non_randori and randori_start_s > current_cursor_s:
                segments.extend(
                    self._build_distributed_segments(
                        pending_non_randori,
                        current_cursor_s,
                        randori_start_s,
                    )
                )
            pending_non_randori = []

            block = phase_entry.get("randori_block") or {}
            randori_count = coerce_positive_int(block.get("randori_count")) or 1
            randori_duration_s = coerce_positive_seconds(block.get("randori_duration_min"))
            rest_s = coerce_positive_seconds(block.get("rest_between_randoris_min")) or float(default_recovery_min) * 60.0
            rest_s = max(rest_s, self.MIN_SEGMENT_DURATION_S)

            phase_cursor_s = randori_start_s
            generated_count = 0
            for randori_index in range(randori_count):
                if phase_cursor_s >= total_duration_s:
                    break

                current_randori_start_s = phase_cursor_s
                if randori_duration_s is not None:
                    current_randori_end_s = min(total_duration_s, current_randori_start_s + max(randori_duration_s, self.MIN_SEGMENT_DURATION_S))
                else:
                    current_randori_end_s = min(
                        max(
                            self._detect_recovery_start(hr_sorted, current_randori_start_s),
                            current_randori_start_s + self.MIN_SEGMENT_DURATION_S,
                        ),
                        total_duration_s,
                    )
                if current_randori_end_s <= current_randori_start_s:
                    current_randori_end_s = min(total_duration_s, current_randori_start_s + self.MIN_SEGMENT_DURATION_S)
                if current_randori_end_s <= current_randori_start_s:
                    break

                segments.append(
                    {
                        "label": "randori",
                        "start_offset_s": current_randori_start_s,
                        "end_offset_s": current_randori_end_s,
                        "source": "auto",
                        "phase_uid": phase_entry.get("phase_uid"),
                        "locked": False,
                    }
                )
                generated_count += 1
                phase_cursor_s = current_randori_end_s

                if randori_index >= randori_count - 1:
                    continue

                recovery_end_s = min(total_duration_s, phase_cursor_s + rest_s)
                if recovery_end_s <= phase_cursor_s:
                    break
                segments.append(
                    {
                        "label": "recuperation",
                        "start_offset_s": phase_cursor_s,
                        "end_offset_s": recovery_end_s,
                        "source": "auto",
                        "phase_uid": phase_entry.get("phase_uid"),
                        "locked": False,
                    }
                )
                phase_cursor_s = recovery_end_s

            if generated_count == 0:
                raise ValueError("Impossible d'auto-generer : la phase randoris ne produit aucun segment exploitable.")
            current_cursor_s = phase_cursor_s

        if pending_non_randori:
            segments.extend(self._build_distributed_segments(pending_non_randori, current_cursor_s, total_duration_s))

        normalized_segments = self.normalize_fc_phase_segments(segments, total_duration_s)
        if not normalized_segments:
            raise ValueError("Aucun segment temporel n'a pu etre genere pour cette seance.")
        normalized_segments[-1]["end_offset_s"] = round(total_duration_s, 3)
        normalized_segments[-1]["duration_s"] = round(normalized_segments[-1]["end_offset_s"] - normalized_segments[-1]["start_offset_s"], 3)
        return self.normalize_fc_phase_segments(normalized_segments, total_duration_s)

    def _build_judo_phase_entries(self, session: ProcessedSession) -> list[dict[str, Any]]:
        randori_blocks_by_uid = {
            block.get("phase_uid"): block
            for block in (session.judo_randori_blocks or [])
            if isinstance(block, dict) and block.get("phase_uid")
        }
        entries: list[dict[str, Any]] = []
        for phase in session.judo_phases or []:
            if not isinstance(phase, dict):
                continue
            raw_label = str(phase.get("phase_label") or phase.get("label") or "").strip().lower()
            if not raw_label:
                continue
            label = self._map_judo_phase_to_segment_label(raw_label)
            if label is None:
                continue
            phase_uid = phase.get("phase_uid")
            entries.append(
                {
                    "label": label,
                    "phase_uid": phase_uid,
                    "randori_block": randori_blocks_by_uid.get(phase_uid),
                }
            )
        return entries

    @staticmethod
    def _map_judo_phase_to_segment_label(raw_label: str) -> str | None:
        if raw_label.startswith("randoris"):
            return "randori"
        if raw_label == "echauffement":
            return "echauffement"
        if raw_label == "technique":
            return "technique"
        if raw_label in {"retour_calme", "retour calme"}:
            return "retour_calme"
        if raw_label in {"autres", "autre"}:
            return "autre"
        return None

    def _detect_randori_start(self, hr_sorted: pd.DataFrame, cursor_s: float, threshold_bpm: float) -> float:
        filtered = hr_sorted.loc[hr_sorted["t_offset_s"] >= max(cursor_s, 0.0)].copy()
        if filtered.empty:
            raise ValueError("Impossible d'auto-generer : aucune donnee FC disponible apres la position courante.")
        filtered["bpm_prev_10s"] = filtered["bpm"].shift(10)
        filtered["rise_10s"] = filtered["bpm"] - filtered["bpm_prev_10s"]
        rapid_rise = filtered.loc[(filtered["bpm"].fillna(0) >= threshold_bpm) & (filtered["rise_10s"].fillna(0) >= 8.0)]
        if not rapid_rise.empty:
            return float(rapid_rise.iloc[0]["t_offset_s"])
        above_threshold = filtered.loc[filtered["bpm"].fillna(0) >= threshold_bpm]
        if above_threshold.empty:
            raise ValueError("Aucun passage FC au-dessus de 160 bpm n'a ete detecte pour demarrer un randori.")
        return float(above_threshold.iloc[0]["t_offset_s"])

    def _detect_recovery_start(self, hr_sorted: pd.DataFrame, randori_start_s: float) -> float:
        filtered = hr_sorted.loc[hr_sorted["t_offset_s"] >= randori_start_s].copy()
        if filtered.empty:
            return randori_start_s + self.MIN_SEGMENT_DURATION_S
        filtered["peak_since_start"] = filtered["bpm"].cummax()
        filtered["drop_since_peak"] = filtered["peak_since_start"] - filtered["bpm"]
        recovery_candidates = filtered.loc[(filtered["bpm"].fillna(999) <= 140.0) & (filtered["drop_since_peak"].fillna(0) >= 12.0)]
        if not recovery_candidates.empty:
            return float(recovery_candidates.iloc[0]["t_offset_s"])
        fallback = filtered.loc[filtered["bpm"].fillna(999) <= 140.0]
        if not fallback.empty:
            return float(fallback.iloc[0]["t_offset_s"])
        return min(float(filtered.iloc[-1]["t_offset_s"]), randori_start_s + 180.0)

    def _build_distributed_segments(
        self,
        phase_entries: list[dict[str, Any]],
        start_s: float,
        end_s: float,
    ) -> list[dict[str, Any]]:
        if end_s <= start_s:
            return []
        items = list(phase_entries)
        if not items:
            fallback_label = "retour_calme" if start_s > 0 else "echauffement"
            items = [{"label": fallback_label, "phase_uid": None}]
        window_s = end_s - start_s
        step_s = window_s / float(len(items))
        segments: list[dict[str, Any]] = []
        cursor_s = start_s
        for index, item in enumerate(items):
            next_cursor_s = end_s if index == len(items) - 1 else cursor_s + step_s
            segments.append(
                {
                    "label": str(item.get("label") or "autre"),
                    "start_offset_s": cursor_s,
                    "end_offset_s": next_cursor_s,
                    "source": "auto",
                    "phase_uid": item.get("phase_uid"),
                    "locked": False,
                }
            )
            cursor_s = next_cursor_s
        return segments

    def normalize_fc_phase_segments(self, segments: list[dict[str, Any]] | None, total_duration_s: float) -> list[dict[str, Any]]:
        if not segments:
            return []

        normalized_rows: list[dict[str, Any]] = []
        for raw_segment in segments:
            if not isinstance(raw_segment, dict):
                continue
            label = str(raw_segment.get("label") or "technique")
            if label not in self.TEMPORAL_SEGMENT_LABELS:
                continue
            start_s = float(max(raw_segment.get("start_offset_s", 0.0), 0.0))
            end_s = float(min(raw_segment.get("end_offset_s", total_duration_s), total_duration_s))
            normalized_rows.append(
                {
                    "label": label,
                    "start_offset_s": start_s,
                    "end_offset_s": end_s,
                    "source": str(raw_segment.get("source") or "manual"),
                    "phase_uid": raw_segment.get("phase_uid"),
                    "locked": bool(raw_segment.get("locked", False)),
                }
            )

        normalized_rows.sort(key=lambda row: (row["start_offset_s"], row["end_offset_s"], row["label"]))
        previous_end_s = 0.0
        final_segments: list[dict[str, Any]] = []
        for index, row in enumerate(normalized_rows):
            start_s = min(max(row["start_offset_s"], 0.0), total_duration_s)
            if final_segments and start_s < previous_end_s:
                start_s = previous_end_s
            end_s = min(max(row["end_offset_s"], start_s + self.MIN_SEGMENT_DURATION_S), total_duration_s)
            if end_s <= start_s:
                end_s = min(total_duration_s, start_s + self.MIN_SEGMENT_DURATION_S)
            if end_s <= start_s:
                continue
            final_segments.append(
                {
                    "segment_index": index,
                    "label": row["label"],
                    "start_offset_s": round(start_s, 3),
                    "end_offset_s": round(end_s, 3),
                    "duration_s": round(end_s - start_s, 3),
                    "source": row["source"],
                    "phase_uid": row["phase_uid"],
                    "locked": row["locked"],
                }
            )
            previous_end_s = end_s
        return final_segments

    def normalize_fc_phase_segments_for_edit(
        self,
        segments: list[dict[str, Any]] | None,
        total_duration_s: float,
        *,
        anchor_index: int | None = None,
    ) -> list[dict[str, Any]]:
        normalized = self.normalize_fc_phase_segments(segments, total_duration_s)
        if not normalized:
            return []
        if anchor_index is None or not (0 <= int(anchor_index) < len(normalized)):
            return normalized

        anchor_index = int(anchor_index)
        adjusted = [dict(segment) for segment in normalized]
        adjusted[anchor_index]["start_offset_s"] = max(0.0, min(float(adjusted[anchor_index]["start_offset_s"]), total_duration_s))
        adjusted[anchor_index]["end_offset_s"] = min(
            max(float(adjusted[anchor_index]["end_offset_s"]), float(adjusted[anchor_index]["start_offset_s"]) + self.MIN_SEGMENT_DURATION_S),
            total_duration_s,
        )
        adjusted[anchor_index]["duration_s"] = round(
            float(adjusted[anchor_index]["end_offset_s"]) - float(adjusted[anchor_index]["start_offset_s"]),
            3,
        )

        for index in range(anchor_index - 1, -1, -1):
            next_start = float(adjusted[index + 1]["start_offset_s"])
            duration = max(float(adjusted[index].get("duration_s", 0.0)), self.MIN_SEGMENT_DURATION_S)
            end_s = min(next_start, total_duration_s)
            start_s = max(0.0, end_s - duration)
            if end_s - start_s < self.MIN_SEGMENT_DURATION_S:
                start_s = max(0.0, end_s - self.MIN_SEGMENT_DURATION_S)
            adjusted[index]["end_offset_s"] = round(end_s, 3)
            adjusted[index]["start_offset_s"] = round(start_s, 3)
            adjusted[index]["duration_s"] = round(end_s - start_s, 3)

        for index in range(anchor_index + 1, len(adjusted)):
            previous_end = float(adjusted[index - 1]["end_offset_s"])
            duration = max(float(adjusted[index].get("duration_s", 0.0)), self.MIN_SEGMENT_DURATION_S)
            start_s = max(previous_end, 0.0)
            end_s = min(start_s + duration, total_duration_s)
            if end_s - start_s < self.MIN_SEGMENT_DURATION_S:
                end_s = min(total_duration_s, start_s + self.MIN_SEGMENT_DURATION_S)
            adjusted[index]["start_offset_s"] = round(start_s, 3)
            adjusted[index]["end_offset_s"] = round(end_s, 3)
            adjusted[index]["duration_s"] = round(end_s - start_s, 3)

        return self.normalize_fc_phase_segments(adjusted, total_duration_s)

    def validate_fc_phase_segments(self, segments: list[dict[str, Any]] | None, total_duration_s: float) -> bool:
        if not segments:
            return False
        previous_end_s = -1.0
        for segment in segments:
            start_s = float(segment.get("start_offset_s", -1.0))
            end_s = float(segment.get("end_offset_s", -1.0))
            duration_s = float(segment.get("duration_s", end_s - start_s))
            if start_s < 0 or end_s > total_duration_s + 0.001:
                return False
            if end_s <= start_s or duration_s <= 0:
                return False
            if start_s < previous_end_s - 0.001:
                return False
            previous_end_s = end_s
        return True

    def save_fc_phase_segments(self, session_id: str, segments: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        session = self.get_sessions([session_id], include_archived=True)[0]
        if not self._is_judo_randoris_session(session):
            raise ValueError("Le flag et les segments temporels sont reserves aux seances judo randoris.")

        meta = self._read_session_meta(session_id)
        total_duration_s = self._session_total_duration_seconds(session)
        normalized_segments = self.normalize_fc_phase_segments(segments or [], total_duration_s)
        is_valid = self.validate_fc_phase_segments(normalized_segments, total_duration_s)

        meta["fc_phase_segments"] = normalized_segments or None
        meta["is_temporally_annotated"] = bool(is_valid and normalized_segments)
        if meta["is_temporally_annotated"]:
            meta["temporally_annotated_at"] = meta.get("temporally_annotated_at") or datetime.now().isoformat()
        else:
            meta["temporally_annotated_at"] = None
        meta["updated_at"] = datetime.now().isoformat()
        self._write_session_meta(session_id, meta)
        return normalized_segments

    def save_fc_phase_segments_for_edit(
        self,
        session_id: str,
        segments: list[dict[str, Any]] | None,
        *,
        anchor_index: int | None = None,
    ) -> list[dict[str, Any]]:
        session = self.get_sessions([session_id], include_archived=True)[0]
        total_duration_s = self._session_total_duration_seconds(session)
        normalized_segments = self.normalize_fc_phase_segments_for_edit(
            segments or [],
            total_duration_s,
            anchor_index=anchor_index,
        )
        return self.save_fc_phase_segments(session_id, normalized_segments)

    def build_fc_phase_segments_export_frame(self, segments: list[dict[str, Any]] | None) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for index, segment in enumerate(segments or []):
            start_s = int(round(float(segment.get("start_offset_s", 0.0))))
            end_s = int(round(float(segment.get("end_offset_s", 0.0))))
            rows.append(
                {
                    "segment_id": int(segment.get("segment_index", index)),
                    "type": str(segment.get("label") or "autre"),
                    "t_debut_s": start_s,
                    "t_fin_s": end_s,
                    "duree_s": max(end_s - start_s, 0),
                }
            )
        return pd.DataFrame(rows)

    def summarize_fc_phase_segments(self, segments: list[dict[str, Any]] | None) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for segment in segments or []:
            rows.append(
                {
                    "ordre": int(segment.get("segment_index", len(rows))),
                    "phase": segment.get("label"),
                    "debut_s": round(float(segment.get("start_offset_s", 0.0)), 1),
                    "fin_s": round(float(segment.get("end_offset_s", 0.0)), 1),
                    "duree_s": round(float(segment.get("duration_s", 0.0)), 1),
                    "origine": segment.get("source", "manual"),
                }
            )
        return pd.DataFrame(rows)

    def delete_session(self, session_id: str) -> None:
        for branch in ("raw", "processed"):
            target_dir = os.path.join(self.output_dir, branch, session_id)
            if os.path.isdir(target_dir):
                shutil.rmtree(target_dir)

    def _meta_path(self, session_id: str) -> str:
        return os.path.join(self.processed_root, session_id, "session_meta.json")

    def _read_session_meta(self, session_id: str) -> dict[str, Any]:
        meta_path = self._meta_path(session_id)
        with open(meta_path, encoding="utf-8-sig") as file_obj:
            return json.load(file_obj)

    def _write_session_meta(self, session_id: str, meta: dict[str, Any]) -> None:
        meta_path = self._meta_path(session_id)
        with open(meta_path, "w", encoding="utf-8") as file_obj:
            json.dump(meta, file_obj, ensure_ascii=False, indent=2)

    def _session_total_duration_seconds(self, session: ProcessedSession, hr_frame: pd.DataFrame | None = None) -> float:
        if hr_frame is not None and not hr_frame.empty and "t_offset_ms" in hr_frame.columns:
            return max(float(hr_frame["t_offset_ms"].max()) / 1000.0, float(session.duree_s or 0.0))
        return float(session.duree_s or 0.0)

    @staticmethod
    def _normalize_optional_int(value: Any) -> int | None:
        if value in (None, "", "NA"):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _normalize_dict_list(value: Any) -> list[dict[str, Any]] | None:
        if not value:
            return None
        items: list[dict[str, Any]] = []
        for entry in value:
            if isinstance(entry, dict):
                items.append(dict(entry))
        return items or None

    @staticmethod
    def _is_judo_randoris_session(session: ProcessedSession) -> bool:
        return session.activity_family == "judo" and session.judo_session_type == "randoris"

    @staticmethod
    def get_session_start(session: ProcessedSession) -> datetime | None:
        if session.fc_start_ts:
            return datetime.fromisoformat(session.fc_start_ts)
        if session.date and session.heure_debut:
            return datetime.fromisoformat(f"{session.date}T{session.heure_debut}")
        return None

    @classmethod
    def get_session_end(cls, session: ProcessedSession) -> datetime | None:
        start_dt = cls.get_session_start(session)
        if start_dt is None:
            return None
        return start_dt + timedelta(seconds=float(session.duree_s or 0.0))

    def find_overlapping_sessions(
        self,
        start_dt: datetime,
        end_dt: datetime,
        ignore_session_ids: set[str] | None = None,
    ) -> list[ProcessedSession]:
        overlaps: list[ProcessedSession] = []
        ignored = ignore_session_ids or set()

        for session in self.list_sessions():
            if session.session_id in ignored:
                continue

            session_start = self.get_session_start(session)
            session_end = self.get_session_end(session)
            if session_start is None or session_end is None:
                continue

            if session_start <= end_dt and session_end >= start_dt:
                overlaps.append(session)

        return overlaps