from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from polar_app.models import PolarFileInfo, ProcessedSession
from polar_app.repository import ProcessedSessionRepository

try:
    import pyarrow  # noqa: F401
    PARQUET_AVAILABLE = True
    DEFAULT_EXTENSION = "parquet"
except ImportError:
    try:
        import fastparquet  # noqa: F401
        PARQUET_AVAILABLE = True
        DEFAULT_EXTENSION = "parquet"
    except ImportError:
        PARQUET_AVAILABLE = False
        DEFAULT_EXTENSION = "csv"


@dataclass(slots=True)
class IncompleteSegmentInfo:
    session_id: str
    device_id: str
    missing_types: list[str]
    available_types: list[str]


@dataclass(slots=True)
class DuplicateSegmentInfo:
    kept_session_id: str
    discarded_session_id: str
    reason: str


@dataclass(slots=True)
class ParsedSegment:
    session_id: str
    device_id: str
    rr_path: str
    hr_path: str
    rr_frame: pd.DataFrame
    hr_frame: pd.DataFrame
    t_start: datetime
    t_end: datetime
    duration_s: float
    rr_count: int
    hr_count: int


@dataclass(slots=True)
class ActivityPlan:
    activity_index: int
    activity_id: str
    device_id: str
    annotation_default: str
    segments: list[ParsedSegment]
    rr_frame: pd.DataFrame
    hr_frame: pd.DataFrame
    t_start: datetime
    t_end: datetime
    duration_s: float
    gaps_s: list[int]
    gap_reviews: list[dict]
    duplicate_segments: list[DuplicateSegmentInfo]
    overlapping_sessions: list[ProcessedSession]


@dataclass(slots=True)
class ImportPlan:
    total_txt_files: int
    ignored_files: list[str]
    incomplete_segments: list[IncompleteSegmentInfo]
    duplicate_segments: list[DuplicateSegmentInfo]
    activities: list[ActivityPlan]
    pending_gap_reviews: list[dict]
    force_single_activity: bool
    complete_segments_count: int
    parse_errors: list[str]


@dataclass(slots=True)
class ImportExecutionResult:
    imported_session_ids: list[str]
    overwritten_session_ids: list[str]
    skipped_activity_ids: list[str]
    deleted_source_files: list[str]


class PolarImporter:
    POLAR_PATTERN = re.compile(
        r"Polar_H10_([A-Z0-9]+)_(\d{8})_(\d{6})_(HR|RR)\.txt$",
        re.IGNORECASE,
    )
    FILENAME_PAIR_TOLERANCE_SECONDS = 1
    FUSION_AUTO_MAX_S = 30
    FUSION_CONFIRM_MAX_S = 90
    RR_MEDIAN_FALLBACK_MS = 500.0
    WU_COURT_S = 5.0
    WU_LONG_S = 8.0
    SEUIL_GAP_DUREE_S = 15.0

    def __init__(self, source_dir: str, output_dir: str) -> None:
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.extension = DEFAULT_EXTENSION

    @staticmethod
    def default_annotation(meta: dict) -> str:
        return f"Activite {meta['heure_debut'][:5]}"

    @classmethod
    def prompt_annotation(cls, meta: dict) -> str:
        default_annotation = cls.default_annotation(meta)
        prompt = (
            f"  Annotation pour l'activite {meta['session_id']} "
            f"({meta['date']} {meta['heure_debut']}) [{default_annotation}] : "
        )
        annotation = input(prompt).strip()
        return annotation or default_annotation

    def parse_filename(self, filename: str) -> PolarFileInfo | None:
        match = self.POLAR_PATTERN.match(os.path.basename(filename))
        if not match:
            return None
        device_id, date_str, time_str, file_type = match.groups()
        return PolarFileInfo(
            device_id=device_id,
            session_id=f"{date_str}_{time_str}",
            date=f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
            heure=f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}",
            file_type=file_type.upper(),
            filepath=filename,
        )

    @staticmethod
    def _session_id_to_datetime(session_id: str) -> datetime | None:
        try:
            return datetime.strptime(session_id, "%Y%m%d_%H%M%S")
        except ValueError:
            return None

    @staticmethod
    def _make_gap_decision_key(previous_segment: ParsedSegment, current_segment: ParsedSegment) -> str:
        return f"{previous_segment.session_id}->{current_segment.session_id}"

    @staticmethod
    def _gap_review_dict(previous_segment: ParsedSegment, current_segment: ParsedSegment, gap_s: int, status: str, merge: bool, requires_confirmation: bool) -> dict:
        return {
            "decision_key": PolarImporter._make_gap_decision_key(previous_segment, current_segment),
            "prev_session_id": previous_segment.session_id,
            "next_session_id": current_segment.session_id,
            "gap_s": int(gap_s),
            "status": status,
            "merge": bool(merge),
            "requires_confirmation": bool(requires_confirmation),
            "start_prev": previous_segment.t_end.isoformat(),
            "start_next": current_segment.t_start.isoformat(),
        }

    def _reconcile_nearby_partial_segments(self, grouped_files: dict[str, dict]) -> dict[str, dict]:
        reconciled = {session_id: dict(content) for session_id, content in grouped_files.items()}
        partial_session_ids = [
            session_id
            for session_id, content in reconciled.items()
            if ("RR" in content) ^ ("HR" in content)
        ]
        consumed: set[str] = set()

        for session_id in sorted(partial_session_ids):
            if session_id in consumed or session_id not in reconciled:
                continue

            content = reconciled[session_id]
            missing_type = "RR" if "RR" not in content else "HR"
            current_dt = self._session_id_to_datetime(session_id)
            if current_dt is None:
                continue

            best_candidate_id = None
            best_delta_seconds = None

            for candidate_id in partial_session_ids:
                if candidate_id == session_id or candidate_id in consumed or candidate_id not in reconciled:
                    continue

                candidate = reconciled[candidate_id]
                if missing_type not in candidate:
                    continue
                if content.get("device_id") != candidate.get("device_id"):
                    continue

                candidate_dt = self._session_id_to_datetime(candidate_id)
                if candidate_dt is None:
                    continue

                delta_seconds = abs((candidate_dt - current_dt).total_seconds())
                if delta_seconds > self.FILENAME_PAIR_TOLERANCE_SECONDS:
                    continue

                if best_delta_seconds is None or delta_seconds < best_delta_seconds:
                    best_candidate_id = candidate_id
                    best_delta_seconds = delta_seconds

            if best_candidate_id is None:
                continue

            candidate = reconciled[best_candidate_id]
            canonical_session_id = min(session_id, best_candidate_id)
            merged = dict(content if canonical_session_id == session_id else candidate)
            merged["device_id"] = content.get("device_id") or candidate.get("device_id", "")
            for file_type in ("RR", "HR"):
                if file_type in content:
                    merged[file_type] = content[file_type]
                if file_type in candidate:
                    merged[file_type] = candidate[file_type]

            reconciled.pop(session_id, None)
            reconciled.pop(best_candidate_id, None)
            reconciled[canonical_session_id] = merged
            consumed.add(session_id)
            consumed.add(best_candidate_id)

        return reconciled

    def read_rr(self, filepath: str) -> pd.DataFrame:
        rows: list[dict] = []
        with open(filepath, encoding="utf-8-sig") as file_obj:
            lines = file_obj.read().strip().splitlines()
        if len(lines) <= 1:
            raise ValueError(f"fichier RR vide ou sans mesures dans {filepath}")
        for line in lines[1:]:
            parts = [part.strip() for part in line.strip().split(";")]
            if len(parts) < 2 or not parts[0] or not parts[1]:
                continue
            try:
                timestamp = datetime.fromisoformat(parts[0])
                rr_value = int(parts[1])
            except ValueError:
                continue
            rows.append(
                {
                    "timestamp": timestamp,
                    "rr_interval_ms": rr_value,
                    "bpm_instantane": round(60000 / rr_value) if rr_value > 0 else 0,
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError(f"Aucune donnee RR exploitable dans {filepath}")
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        frame["rr_interval_ms"] = frame["rr_interval_ms"].astype("int32")
        frame["bpm_instantane"] = frame["bpm_instantane"].astype("int16")
        return frame

    def read_hr(self, filepath: str) -> pd.DataFrame:
        rows: list[dict] = []
        with open(filepath, encoding="utf-8-sig") as file_obj:
            lines = file_obj.read().strip().splitlines()
        if len(lines) <= 1:
            return pd.DataFrame(columns=["timestamp", "bpm", "breathing_rpm"])
        for line in lines[1:]:
            parts = [part.strip() for part in line.strip().split(";")]
            if len(parts) < 2 or not parts[0] or not parts[1]:
                continue
            try:
                timestamp = datetime.fromisoformat(parts[0])
                bpm_value = int(parts[1])
            except ValueError:
                continue
            breathing = None
            for breathing_index in (3, 2):
                if len(parts) > breathing_index and parts[breathing_index]:
                    try:
                        breathing = float(parts[breathing_index].replace(",", "."))
                    except ValueError:
                        breathing = None
                    break
            rows.append({"timestamp": timestamp, "bpm": bpm_value, "breathing_rpm": breathing})
        if not rows:
            return pd.DataFrame(columns=["timestamp", "bpm", "breathing_rpm"])
        frame = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        frame["bpm"] = frame["bpm"].astype("int16")
        return frame

    def plan_import(
        self,
        gap_threshold_seconds: int | None = None,
        post_reconnect_flag_duration_seconds: int | None = None,
        force_single_activity: bool = False,
        gap_merge_decisions: dict[str, bool] | None = None,
    ) -> ImportPlan:
        grouped_files: dict[str, dict] = defaultdict(dict)
        ignored_files: list[str] = []
        incomplete_segments: list[IncompleteSegmentInfo] = []
        duplicate_segments: list[DuplicateSegmentInfo] = []
        parsed_segments: list[ParsedSegment] = []
        parse_errors: list[str] = []
        total_txt_files = 0
        pending_gap_reviews: list[dict] = []

        if not self.source_dir or not os.path.isdir(self.source_dir):
            return ImportPlan(0, [], [], [], [], [], force_single_activity, 0, [])

        for filename in sorted(os.listdir(self.source_dir)):
            if not filename.lower().endswith(".txt"):
                continue
            total_txt_files += 1
            file_info = self.parse_filename(filename)
            if file_info is None:
                ignored_files.append(filename)
                continue
            session_entry = grouped_files[file_info.session_id]
            session_entry[file_info.file_type] = os.path.join(self.source_dir, filename)
            session_entry["device_id"] = file_info.device_id

        grouped_files = self._reconcile_nearby_partial_segments(grouped_files)

        for session_id, content in sorted(grouped_files.items()):
            missing_types = [name for name in ("RR", "HR") if name not in content]
            if missing_types:
                incomplete_segments.append(
                    IncompleteSegmentInfo(session_id, content.get("device_id", ""), missing_types, sorted(name for name in ("RR", "HR") if name in content))
                )
                continue
            try:
                rr_frame = self.read_rr(content["RR"])
                hr_frame = self.read_hr(content["HR"])
            except ValueError as exc:
                parse_errors.append(f"{session_id} : {exc}")
                continue

            if hr_frame.empty:
                parse_errors.append(f"{session_id} : aucune donnee HR exploitable dans {content['HR']}")
                continue

            t_start = rr_frame["timestamp"].min().to_pydatetime()
            t_end = rr_frame["timestamp"].max().to_pydatetime()
            parsed_segments.append(
                ParsedSegment(
                    session_id=session_id,
                    device_id=content["device_id"],
                    rr_path=content["RR"],
                    hr_path=content["HR"],
                    rr_frame=rr_frame,
                    hr_frame=hr_frame,
                    t_start=t_start,
                    t_end=t_end,
                    duration_s=max((t_end - t_start).total_seconds(), 0.0),
                    rr_count=len(rr_frame),
                    hr_count=len(hr_frame),
                )
            )

        kept_by_device: dict[str, list[ParsedSegment]] = defaultdict(list)
        for device_id, device_segments in self._group_segments_by_device(parsed_segments).items():
            kept_segments, duplicates = self._deduplicate_segments(device_segments)
            kept_by_device[device_id].extend(kept_segments)
            duplicate_segments.extend(duplicates)

        existing_sessions = ProcessedSessionRepository(self.output_dir).list_sessions()
        activities: list[ActivityPlan] = []
        used_activity_ids: set[str] = set()
        decisions = gap_merge_decisions or {}

        for device_id, kept_segments in sorted(kept_by_device.items()):
            ordered_segments = sorted(kept_segments, key=lambda item: item.t_start)
            segment_groups, group_gap_reviews, group_pending = self._build_activity_groups(ordered_segments, decisions, force_single_activity)
            pending_gap_reviews.extend(group_pending)
            for group, merged_gap_reviews in zip(segment_groups, group_gap_reviews):
                if not group:
                    continue
                rr_frame, hr_frame, gaps_s = self._make_activity_frames(group, merged_gap_reviews)
                activity_id = self._make_unique_activity_id(group[0].t_start, used_activity_ids)
                session_ids = {segment.session_id for segment in group}
                related_duplicates = [
                    duplicate for duplicate in duplicate_segments
                    if duplicate.kept_session_id in session_ids or duplicate.discarded_session_id in session_ids
                ]
                meta = {
                    "session_id": activity_id,
                    "date": group[0].t_start.strftime("%Y-%m-%d"),
                    "heure_debut": group[0].t_start.strftime("%H:%M:%S"),
                }
                activities.append(
                    ActivityPlan(
                        activity_index=len(activities) + 1,
                        activity_id=activity_id,
                        device_id=device_id,
                        annotation_default=self.default_annotation(meta),
                        segments=group,
                        rr_frame=rr_frame,
                        hr_frame=hr_frame,
                        t_start=group[0].t_start,
                        t_end=group[-1].t_end,
                        duration_s=max((group[-1].t_end - group[0].t_start).total_seconds(), 0.0),
                        gaps_s=gaps_s,
                        gap_reviews=merged_gap_reviews,
                        duplicate_segments=related_duplicates,
                        overlapping_sessions=self._find_existing_overlaps(existing_sessions, group[0].t_start, group[-1].t_end),
                    )
                )

        activities.sort(key=lambda activity: activity.t_start)
        for index, activity in enumerate(activities, start=1):
            activity.activity_index = index

        return ImportPlan(
            total_txt_files=total_txt_files,
            ignored_files=ignored_files,
            incomplete_segments=incomplete_segments,
            duplicate_segments=duplicate_segments,
            activities=activities,
            pending_gap_reviews=pending_gap_reviews,
            force_single_activity=force_single_activity,
            complete_segments_count=len(parsed_segments),
            parse_errors=parse_errors,
        )

    def _build_activity_groups(
        self,
        ordered_segments: list[ParsedSegment],
        gap_merge_decisions: dict[str, bool],
        force_single_activity: bool,
    ) -> tuple[list[list[ParsedSegment]], list[list[dict]], list[dict]]:
        if not ordered_segments:
            return [], [], []

        groups: list[list[ParsedSegment]] = [[ordered_segments[0]]]
        group_gap_reviews: list[list[dict]] = [[]]
        pending_reviews: list[dict] = []

        for current_segment in ordered_segments[1:]:
            previous_segment = groups[-1][-1]
            gap_s = max(int(round((current_segment.t_start - previous_segment.t_end).total_seconds())), 0)
            decision_key = self._make_gap_decision_key(previous_segment, current_segment)

            if force_single_activity:
                merge = True
                status = "forced_merge"
                requires_confirmation = False
            elif gap_s < self.FUSION_AUTO_MAX_S:
                merge = True
                status = "auto_merge"
                requires_confirmation = False
            elif gap_s <= self.FUSION_CONFIRM_MAX_S:
                requires_confirmation = True
                if decision_key in gap_merge_decisions:
                    merge = bool(gap_merge_decisions[decision_key])
                    status = "user_confirmed_merge" if merge else "user_rejected_merge"
                else:
                    merge = False
                    status = "pending_confirmation"
            else:
                merge = False
                status = "split_long_gap"
                requires_confirmation = False

            review = self._gap_review_dict(previous_segment, current_segment, gap_s, status, merge, requires_confirmation)
            if requires_confirmation and status == "pending_confirmation":
                pending_reviews.append(review)

            if merge:
                groups[-1].append(current_segment)
                group_gap_reviews[-1].append(review)
            else:
                groups.append([current_segment])
                group_gap_reviews.append([])

        return groups, group_gap_reviews, pending_reviews

    def execute_import_plan(
        self,
        plan: ImportPlan,
        annotation_overrides: dict[str, str] | None = None,
        overwrite_activity_ids: set[str] | None = None,
        delete_source_files: bool = True,
    ) -> ImportExecutionResult:
        overwrite_activity_ids = overwrite_activity_ids or set()
        imported_session_ids: list[str] = []
        overwritten_session_ids: list[str] = []
        skipped_activity_ids: list[str] = []
        deleted_source_files: list[str] = []
        removed_existing_ids: set[str] = set()
        source_files_to_delete: set[str] = set()

        for activity in plan.activities:
            if activity.overlapping_sessions and activity.activity_id not in overwrite_activity_ids:
                skipped_activity_ids.append(activity.activity_id)
                continue
            for session in activity.overlapping_sessions:
                if session.session_id in removed_existing_ids:
                    continue
                self._remove_existing_session(session.session_id)
                removed_existing_ids.add(session.session_id)
                overwritten_session_ids.append(session.session_id)
            annotation = activity.annotation_default
            if annotation_overrides and annotation_overrides.get(activity.activity_id):
                annotation = annotation_overrides[activity.activity_id].strip() or activity.annotation_default
            self._save_activity(activity, annotation, plan)
            imported_session_ids.append(activity.activity_id)
            for segment in activity.segments:
                source_files_to_delete.add(segment.rr_path)
                source_files_to_delete.add(segment.hr_path)

        if delete_source_files:
            deleted_source_files = self._delete_source_files(sorted(source_files_to_delete))

        return ImportExecutionResult(imported_session_ids, overwritten_session_ids, skipped_activity_ids, deleted_source_files)

    def import_sessions(
        self,
        annotation_overrides: dict[str, str] | None = None,
        prompt_for_annotation: bool = True,
        gap_threshold_seconds: int | None = None,
        post_reconnect_flag_duration_seconds: int | None = None,
        force_single_activity: bool = False,
        overwrite_existing: bool = False,
        gap_merge_decisions: dict[str, bool] | None = None,
    ) -> list[str]:
        plan = self.plan_import(gap_threshold_seconds, post_reconnect_flag_duration_seconds, force_single_activity, gap_merge_decisions)
        if not plan.activities:
            print("Aucune activite PSL exploitable detectee.")
            return []
        self._print_plan_summary(plan)
        resolved_annotations: dict[str, str] = {}
        for activity in plan.activities:
            meta = {
                "session_id": activity.activity_id,
                "date": activity.t_start.strftime("%Y-%m-%d"),
                "heure_debut": activity.t_start.strftime("%H:%M:%S"),
            }
            if annotation_overrides and annotation_overrides.get(activity.activity_id):
                resolved_annotations[activity.activity_id] = annotation_overrides[activity.activity_id]
            elif prompt_for_annotation:
                resolved_annotations[activity.activity_id] = self.prompt_annotation(meta)
            else:
                resolved_annotations[activity.activity_id] = self.default_annotation(meta)
        overwrite_ids = set()
        if overwrite_existing:
            overwrite_ids = {activity.activity_id for activity in plan.activities if activity.overlapping_sessions}
        result = self.execute_import_plan(plan, resolved_annotations, overwrite_ids, delete_source_files=True)
        self._print_execution_summary(result)
        self._print_output_tree()
        return result.imported_session_ids

    def reprocess_saved_sessions(self, session_ids: list[str] | None = None) -> dict[str, list[str]]:
        raw_root = os.path.join(self.output_dir, "raw")
        processed_root = os.path.join(self.output_dir, "processed")
        if not os.path.isdir(raw_root):
            return {"regenerated": [], "skipped": session_ids or []}

        repository = ProcessedSessionRepository(self.output_dir)
        existing_sessions = {session.session_id: session for session in repository.list_sessions()}
        selected_ids = sorted(session_ids or [name for name in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, name))])
        regenerated: list[str] = []
        skipped: list[str] = []

        for session_id in selected_ids:
            raw_dir = os.path.join(raw_root, session_id)
            processed_dir = os.path.join(processed_root, session_id)
            if not os.path.isdir(raw_dir):
                skipped.append(session_id)
                continue

            session_meta_path = os.path.join(processed_dir, "session_meta.json")
            session_meta: dict = {}
            if os.path.isfile(session_meta_path):
                with open(session_meta_path, encoding="utf-8-sig") as file_obj:
                    session_meta = json.load(file_obj)
            session = existing_sessions.get(session_id)
            device_id = session.device_id if session else session_meta.get("device_id", "")
            annotation = (session.annotation if session else session_meta.get("annotation")) or session_id

            try:
                segments = self._load_segments_from_raw_session(raw_dir, device_id)
            except ValueError:
                skipped.append(session_id)
                continue

            gap_reviews: list[dict] = []
            for idx in range(1, len(segments)):
                gap_s = max(int(round((segments[idx].t_start - segments[idx - 1].t_end).total_seconds())), 0)
                requires_confirmation = self.FUSION_AUTO_MAX_S <= gap_s <= self.FUSION_CONFIRM_MAX_S
                gap_reviews.append(
                    self._gap_review_dict(
                        segments[idx - 1],
                        segments[idx],
                        gap_s,
                        "reprocess_preserve_raw_group",
                        True,
                        requires_confirmation,
                    )
                )

            rr_frame, hr_frame, gaps_s = self._make_activity_frames(segments, gap_reviews)
            activity = ActivityPlan(
                activity_index=1,
                activity_id=session_id,
                device_id=device_id,
                annotation_default=annotation,
                segments=segments,
                rr_frame=rr_frame,
                hr_frame=hr_frame,
                t_start=segments[0].t_start,
                t_end=segments[-1].t_end,
                duration_s=max((segments[-1].t_end - segments[0].t_start).total_seconds(), 0.0),
                gaps_s=gaps_s,
                gap_reviews=gap_reviews,
                duplicate_segments=[],
                overlapping_sessions=[],
            )
            plan = ImportPlan(0, [], [], [], [activity], [], False, len(segments), [])
            self._save_activity(activity, annotation, plan)
            self._write_reprocess_log(processed_dir, session_id, gap_reviews)
            regenerated.append(session_id)

        return {"regenerated": regenerated, "skipped": skipped}

    def _load_segments_from_raw_session(self, raw_dir: str, device_id: str) -> list[ParsedSegment]:
        segment_pairs: dict[str, dict[str, str]] = defaultdict(dict)
        for filename in sorted(os.listdir(raw_dir)):
            full_path = os.path.join(raw_dir, filename)
            if not os.path.isfile(full_path):
                continue
            match = re.match(r"segment_(\d+)_polar_(RR|HR)\.txt$", filename, re.IGNORECASE)
            if match:
                seg_index, file_type = match.groups()
                segment_pairs[seg_index][file_type.upper()] = full_path

        if not segment_pairs:
            legacy_rr = os.path.join(raw_dir, "polar_RR.txt")
            legacy_hr = os.path.join(raw_dir, "polar_HR.txt")
            if os.path.isfile(legacy_rr) and os.path.isfile(legacy_hr):
                segment_pairs["0"] = {"RR": legacy_rr, "HR": legacy_hr}

        segments: list[ParsedSegment] = []
        for seg_index in sorted(segment_pairs, key=lambda value: int(value)):
            pair = segment_pairs[seg_index]
            if "RR" not in pair or "HR" not in pair:
                continue
            rr_frame = self.read_rr(pair["RR"])
            hr_frame = self.read_hr(pair["HR"])
            if rr_frame.empty or hr_frame.empty:
                continue
            t_start = rr_frame["timestamp"].min().to_pydatetime()
            t_end = rr_frame["timestamp"].max().to_pydatetime()
            segment_session_id = t_start.strftime("%Y%m%d_%H%M%S")
            segments.append(
                ParsedSegment(
                    session_id=segment_session_id,
                    device_id=device_id,
                    rr_path=pair["RR"],
                    hr_path=pair["HR"],
                    rr_frame=rr_frame,
                    hr_frame=hr_frame,
                    t_start=t_start,
                    t_end=t_end,
                    duration_s=max((t_end - t_start).total_seconds(), 0.0),
                    rr_count=len(rr_frame),
                    hr_count=len(hr_frame),
                )
            )
        if not segments:
            raise ValueError(f"Aucun segment raw exploitable dans {raw_dir}")
        return sorted(segments, key=lambda item: item.t_start)

    def save_dataframe(self, frame: pd.DataFrame, path: str) -> None:
        if self.extension == "parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)

    def _group_segments_by_device(self, segments: list[ParsedSegment]) -> dict[str, list[ParsedSegment]]:
        grouped: dict[str, list[ParsedSegment]] = defaultdict(list)
        for segment in segments:
            grouped[segment.device_id].append(segment)
        return grouped

    def _deduplicate_segments(self, segments: list[ParsedSegment]) -> tuple[list[ParsedSegment], list[DuplicateSegmentInfo]]:
        kept: list[ParsedSegment] = []
        duplicates: list[DuplicateSegmentInfo] = []
        for segment in sorted(segments, key=lambda item: (item.t_start, item.t_end)):
            if not kept:
                kept.append(segment)
                continue
            previous = kept[-1]
            if segment.t_start > previous.t_end and segment.t_start != previous.t_start:
                kept.append(segment)
                continue
            preferred, discarded = self._pick_preferred_segment(previous, segment)
            kept[-1] = preferred
            duplicates.append(
                DuplicateSegmentInfo(
                    kept_session_id=preferred.session_id,
                    discarded_session_id=discarded.session_id,
                    reason="timestamp de debut identique" if segment.t_start == previous.t_start else "overlap temporel",
                )
            )
        return kept, duplicates

    @staticmethod
    def _pick_preferred_segment(first: ParsedSegment, second: ParsedSegment) -> tuple[ParsedSegment, ParsedSegment]:
        first_score = (first.duration_s, first.rr_count, first.hr_count)
        second_score = (second.duration_s, second.rr_count, second.hr_count)
        if second_score > first_score:
            return second, first
        return first, second

    def _make_activity_frames(self, segments: list[ParsedSegment], gap_reviews: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
        rr_parts: list[pd.DataFrame] = []
        hr_parts: list[pd.DataFrame] = []
        gaps_s: list[int] = []
        review_by_key = {review["decision_key"]: review for review in gap_reviews}

        for index, segment in enumerate(segments):
            rr_part = segment.rr_frame.copy()
            rr_part["segment_id"] = index
            rr_part["segment_session_id"] = segment.session_id
            rr_part["source_file"] = os.path.basename(segment.rr_path)
            rr_part["pipeline_flag"] = "OK"

            hr_part = segment.hr_frame.copy()
            if not hr_part.empty:
                hr_part["segment_id"] = index
                hr_part["segment_session_id"] = segment.session_id
                hr_part["source_file"] = os.path.basename(segment.hr_path)
                hr_part["pipeline_flag"] = "OK"

            if index > 0:
                previous = segments[index - 1]
                gap_s = max(int(round((segment.t_start - previous.t_end).total_seconds())), 0)
                decision_key = self._make_gap_decision_key(previous, segment)
                review = review_by_key.get(decision_key)
                if review is not None and review.get("merge"):
                    gaps_s.append(gap_s)
                    gap_frame = self._build_gap_frame(previous, segment, gap_s)
                    if not gap_frame.empty:
                        rr_parts.append(gap_frame)
                    rr_part = self._mark_post_reconnect(rr_part, segment.t_start, gap_s)
                    if not hr_part.empty:
                        hr_part = self._mark_post_reconnect(hr_part, segment.t_start, gap_s)

            rr_parts.append(rr_part)
            if not hr_part.empty:
                hr_parts.append(hr_part)

        rr_frame = pd.concat(rr_parts, ignore_index=True) if rr_parts else pd.DataFrame()
        hr_frame = pd.concat(hr_parts, ignore_index=True) if hr_parts else pd.DataFrame()

        if not rr_frame.empty:
            rr_frame = rr_frame.sort_values("timestamp").reset_index(drop=True)
            activity_start = rr_frame.loc[rr_frame["rr_interval_ms"].fillna(0) > 0, "timestamp"].min()
            rr_frame["t_offset_ms"] = ((rr_frame["timestamp"] - activity_start).dt.total_seconds() * 1000).round().astype("int64")
            rr_frame["rr_interval_ms"] = rr_frame["rr_interval_ms"].astype("float64")
            rr_frame["bpm_instantane"] = rr_frame["bpm_instantane"].astype("float64")
            rr_frame = rr_frame[["t_offset_ms", "timestamp", "rr_interval_ms", "bpm_instantane", "segment_id", "segment_session_id", "source_file", "pipeline_flag"]]

        if not hr_frame.empty:
            hr_frame = hr_frame.sort_values("timestamp").reset_index(drop=True)
            activity_start = rr_frame.loc[rr_frame["rr_interval_ms"].fillna(0) > 0, "timestamp"].min()
            hr_frame["t_offset_ms"] = ((hr_frame["timestamp"] - activity_start).dt.total_seconds() * 1000).round().astype("int64")
            hr_frame["bpm"] = hr_frame["bpm"].astype("int16")
            hr_frame = hr_frame[["t_offset_ms", "timestamp", "bpm", "breathing_rpm", "segment_id", "segment_session_id", "source_file", "pipeline_flag"]]
        else:
            hr_frame = pd.DataFrame(columns=["t_offset_ms", "timestamp", "bpm", "breathing_rpm", "segment_id", "segment_session_id", "source_file", "pipeline_flag"])

        return rr_frame, hr_frame, gaps_s

    def _estimate_gap_rr_ms(self, previous_segment: ParsedSegment, next_segment: ParsedSegment) -> float:
        previous_tail = previous_segment.rr_frame["rr_interval_ms"].tail(5)
        next_head = next_segment.rr_frame["rr_interval_ms"].head(5)
        candidates = pd.concat([previous_tail, next_head], ignore_index=True)
        candidates = candidates[candidates > 0]
        if candidates.empty:
            return self.RR_MEDIAN_FALLBACK_MS
        return float(candidates.median())

    def _build_gap_frame(self, previous_segment: ParsedSegment, next_segment: ParsedSegment, gap_s: int) -> pd.DataFrame:
        gap_ms = max(gap_s * 1000.0, 0.0)
        if gap_ms <= 0:
            return pd.DataFrame(columns=["timestamp", "rr_interval_ms", "bpm_instantane", "segment_id", "segment_session_id", "source_file", "pipeline_flag"])

        rr_median_local = self._estimate_gap_rr_ms(previous_segment, next_segment)
        n_placeholders = max(int((gap_ms + max(rr_median_local, 1.0) - 1) // max(rr_median_local, 1.0)), 1)
        step_ms = gap_ms / n_placeholders
        rows: list[dict] = []
        for index in range(n_placeholders):
            timestamp = previous_segment.t_end + timedelta(milliseconds=(index + 1) * step_ms)
            if timestamp >= next_segment.t_start:
                timestamp = next_segment.t_start - timedelta(milliseconds=max(1.0, (n_placeholders - index) * 0.1))
            rows.append(
                {
                    "timestamp": timestamp,
                    "rr_interval_ms": float("nan"),
                    "bpm_instantane": 0.0,
                    "segment_id": -1,
                    "segment_session_id": "",
                    "source_file": "",
                    "pipeline_flag": "GAP",
                }
            )
        return pd.DataFrame(rows)

    def _warmup_flag_for_gap(self, gap_s: float) -> tuple[str, float]:
        if gap_s < self.SEUIL_GAP_DUREE_S:
            return "POST_RECONNECT_COURT", self.WU_COURT_S
        return "POST_RECONNECT_LONG", self.WU_LONG_S

    def _mark_post_reconnect(self, frame: pd.DataFrame, segment_start: datetime, gap_s: int) -> pd.DataFrame:
        if frame.empty:
            return frame
        flag, duration_seconds = self._warmup_flag_for_gap(float(gap_s))
        reconnect_end = segment_start + timedelta(seconds=duration_seconds)
        mask = (frame["timestamp"] <= reconnect_end) & (frame["pipeline_flag"] == "OK")
        frame.loc[mask, "pipeline_flag"] = flag
        return frame

    @staticmethod
    def _find_existing_overlaps(sessions: list[ProcessedSession], start_dt: datetime, end_dt: datetime) -> list[ProcessedSession]:
        overlaps: list[ProcessedSession] = []
        for session in sessions:
            session_start = ProcessedSessionRepository.get_session_start(session)
            session_end = ProcessedSessionRepository.get_session_end(session)
            if session_start is None or session_end is None:
                continue
            if session_start <= end_dt and session_end >= start_dt:
                overlaps.append(session)
        return overlaps

    @staticmethod
    def _make_unique_activity_id(start_dt: datetime, used_ids: set[str]) -> str:
        base_id = start_dt.strftime("%Y%m%d_%H%M%S")
        if base_id not in used_ids:
            used_ids.add(base_id)
            return base_id
        suffix = 2
        while True:
            candidate = f"{base_id}_{suffix:02d}"
            if candidate not in used_ids:
                used_ids.add(candidate)
                return candidate
            suffix += 1

    def _remove_existing_session(self, session_id: str) -> None:
        for branch in ("raw", "processed"):
            target_dir = os.path.join(self.output_dir, branch, session_id)
            if os.path.isdir(target_dir):
                shutil.rmtree(target_dir)

    def _delete_source_files(self, filepaths: list[str]) -> list[str]:
        deleted_files: list[str] = []
        for filepath in filepaths:
            if not filepath or not os.path.isfile(filepath):
                continue
            try:
                os.remove(filepath)
                deleted_files.append(filepath)
            except OSError:
                continue
        return deleted_files

    def _save_activity(self, activity: ActivityPlan, annotation: str, plan: ImportPlan) -> None:
        raw_dir = os.path.join(self.output_dir, "raw", activity.activity_id)
        processed_dir = os.path.join(self.output_dir, "processed", activity.activity_id)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        for segment_index, segment in enumerate(activity.segments):
            rr_target = os.path.join(raw_dir, f"segment_{segment_index:02d}_polar_RR.txt")
            hr_target = os.path.join(raw_dir, f"segment_{segment_index:02d}_polar_HR.txt")
            if os.path.abspath(segment.rr_path) != os.path.abspath(rr_target):
                shutil.copy2(segment.rr_path, rr_target)
            if os.path.abspath(segment.hr_path) != os.path.abspath(hr_target):
                shutil.copy2(segment.hr_path, hr_target)

        rr_out = os.path.join(processed_dir, f"rr.{self.extension}")
        hr_out = os.path.join(processed_dir, f"hr.{self.extension}")
        self.save_dataframe(activity.rr_frame, rr_out)
        self.save_dataframe(activity.hr_frame, hr_out)

        rr_real = activity.rr_frame.loc[activity.rr_frame["rr_interval_ms"].fillna(0) > 0].copy()
        hr_real = activity.hr_frame.copy()
        meta = {
            "session_id": activity.activity_id,
            "device_id": activity.device_id,
            "date": activity.t_start.strftime("%Y-%m-%d"),
            "heure_debut": activity.t_start.strftime("%H:%M:%S"),
            "annotation": annotation,
            "fc_start_ts": activity.t_start.isoformat(),
            "nb_battements_rr": int(len(activity.rr_frame)),
            "nb_points_hr": int(len(activity.hr_frame)),
            "duree_s": float(activity.duration_s),
            "bpm_moyen": round(float(hr_real["bpm"].mean()), 1) if not hr_real.empty else 0.0,
            "bpm_min": int(hr_real["bpm"].min()) if not hr_real.empty else 0,
            "bpm_max": int(hr_real["bpm"].max()) if not hr_real.empty else 0,
            "rr_min_ms": int(rr_real["rr_interval_ms"].min()) if not rr_real.empty else None,
            "rr_max_ms": int(rr_real["rr_interval_ms"].max()) if not rr_real.empty else None,
            "rr_filepath": os.path.join("processed", activity.activity_id, f"rr.{self.extension}"),
            "hr_filepath": os.path.join("processed", activity.activity_id, f"hr.{self.extension}"),
            "format_fichiers": self.extension,
            "import_timestamp": datetime.now().isoformat(),
            "source_session_ids": [segment.session_id for segment in activity.segments],
            "segments_count": len(activity.segments),
            "gaps_count": len(activity.gaps_s),
            "gap_durations_s": activity.gaps_s,
            "duplicates_removed": len(activity.duplicate_segments),
            "incomplete_segments": len(plan.incomplete_segments),
            "import_mode": "pipeline_rr_v2",
            "gap_reviews": activity.gap_reviews,
        }
        with open(os.path.join(processed_dir, "session_meta.json"), "w", encoding="utf-8") as file_obj:
            json.dump(meta, file_obj, ensure_ascii=False, indent=2)
        with open(os.path.join(processed_dir, "import_log.txt"), "w", encoding="utf-8") as file_obj:
            file_obj.write(self._build_activity_log(activity))

    def _write_reprocess_log(self, processed_dir: str, session_id: str, gap_reviews: list[dict]) -> None:
        log_path = os.path.join(processed_dir, "reprocess_log.txt")
        timestamp = datetime.now().isoformat()
        lines = [f"Retraitement pipeline RR : {timestamp}", f"Session : {session_id}"]
        if gap_reviews:
            lines.append("Jonctions conservees depuis raw/ :")
            for review in gap_reviews:
                lines.append(f"  {review['prev_session_id']} -> {review['next_session_id']} | {review['gap_s']}s | {review['status']}")
        with open(log_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("\n".join(lines) + "\n")

    def _build_activity_log(self, activity: ActivityPlan) -> str:
        pipeline_counts = activity.rr_frame["pipeline_flag"].value_counts().to_dict()
        total_lines = len(activity.rr_frame)
        lines = [
            f"=== ACTIVITE {activity.activity_index:03d} ===",
            f"Debut         : {activity.t_start.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Fin           : {activity.t_end.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duree totale  : {self._format_duration(activity.duration_s)}",
            "",
            f"Segments fusionnes : {len(activity.segments)}",
        ]
        for segment_index, segment in enumerate(activity.segments):
            lines.append(f"  Segment {segment_index} : {segment.session_id} | debut {segment.t_start.strftime('%H:%M:%S')} | fin {segment.t_end.strftime('%H:%M:%S')} | duree {self._format_duration(segment.duration_s)}")
        lines.append("")
        lines.append(f"Deconnexions fusionnees : {len(activity.gaps_s)}")
        for review in activity.gap_reviews:
            lines.append(f"  Gap {review['prev_session_id']} -> {review['next_session_id']} | duree {review['gap_s']}s | statut {review['status']}")
        lines.extend(["", f"Doublons detectes : {len(activity.duplicate_segments)}", "", "Pipeline RR :"])
        for flag in sorted(pipeline_counts):
            count = int(pipeline_counts.get(flag, 0))
            percent = (count / total_lines * 100) if total_lines else 0.0
            lines.append(f"  {flag} : {count} ({percent:.1f} %)")
        return "\n".join(lines) + "\n"

    def _print_plan_summary(self, plan: ImportPlan) -> None:
        print("=== RESUME GLOBAL ===")
        print(f"Fichiers txt recus      : {plan.total_txt_files}")
        print(f"Segments complets       : {plan.complete_segments_count}")
        print(f"Segments incomplets     : {len(plan.incomplete_segments)}")
        print(f"Segments invalides      : {len(plan.parse_errors)}")
        print(f"Doublons ecartes        : {len(plan.duplicate_segments)}")
        print(f"Activites detectees     : {len(plan.activities)}")
        print(f"Gaps a confirmer        : {len(plan.pending_gap_reviews)}")
        print()
        for activity in plan.activities:
            gaps_text = " | ".join(f"{gap}s" for gap in activity.gaps_s) if activity.gaps_s else "-"
            print(f"Activite {activity.activity_index} :")
            print(f"  Segments : {len(activity.segments)}")
            print(f"  Gaps     : {gaps_text}")
            print(f"  Duree    : {self._format_duration(activity.duration_s)}")
            print(f"  Sortie   : processed/{activity.activity_id}/")
            if activity.overlapping_sessions:
                print("  Overlaps : " + ", ".join(session.session_id for session in activity.overlapping_sessions))
            print()

    def _print_execution_summary(self, result: ImportExecutionResult) -> None:
        if result.imported_session_ids:
            print("Activites importees : " + ", ".join(result.imported_session_ids))
        if result.overwritten_session_ids:
            print("Activites ecrasees  : " + ", ".join(sorted(set(result.overwritten_session_ids))))
        if result.skipped_activity_ids:
            print("Activites ignorees  : " + ", ".join(result.skipped_activity_ids))
        print()

    def _print_output_tree(self) -> None:
        if not os.path.isdir(self.output_dir):
            return
        print(f"{'=' * 60}")
        print(f"  Import termine -> arborescence creee dans : {self.output_dir}/")
        print(f"{'=' * 60}\n")
        for root, _, files in os.walk(self.output_dir):
            level = root.replace(self.output_dir, "").count(os.sep)
            indent = "  " + "    " * level
            print(f"{indent}{os.path.basename(root)}/")
            file_indent = "  " + "    " * (level + 1)
            for filename in sorted(files):
                size = os.path.getsize(os.path.join(root, filename))
                print(f"{file_indent}{filename} ({self._format_size(size)})")
        print()

    @staticmethod
    def _format_duration(duration_s: float) -> str:
        duration_s = int(round(duration_s))
        minutes, seconds = divmod(duration_s, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours} h {minutes:02d} min {seconds:02d} s"
        return f"{minutes} min {seconds:02d} s"

    @staticmethod
    def _format_size(size_in_bytes: int) -> str:
        if size_in_bytes < 1024:
            return f"{size_in_bytes} o"
        if size_in_bytes < 1024 ** 2:
            return f"{size_in_bytes / 1024:.1f} Ko"
        return f"{size_in_bytes / 1024 ** 2:.2f} Mo"
