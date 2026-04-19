from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

SEGMENT_TYPE_VERSION = 2

OLD_SEGMENT_LABEL_MAP = {
    "randori": "randori_tw",
    "autre": "technique",
}
OLD_PHASE_LABEL_MAP = {
    "randoris tw": "randori_tw",
    "randoris nw pure": "randori_nw",
    "randoris nw": "randori_nw",
    "randoris libres": "randori_tw",
    "echauffement": "echauffement",
    "technique": "technique",
    "retour calme": "retour_calme",
    "retour_calme": "retour_calme",
    "autres": "technique",
    "autre": "technique",
}
OLD_ACTIVITY_LABEL_MAP = {
    "judo > randoris > tw": "judo > randoris > randori_tw",
    "judo > randoris > nw pure": "judo > randoris > randori_nw",
    "judo > randoris > mixtes (nw + tw)": "judo > randoris > randori_tw",
}
OLD_RANDORI_KIND_MAP = {
    "tw": "randori_tw",
    "nw pure": "randori_nw",
    "mixtes (nw + tw)": "randori_tw",
    "libres": "randori_tw",
    "randori": "randori_tw",
    "randori_tw": "randori_tw",
    "randori_nw": "randori_nw",
}


def normalize_segment_label(label: Any) -> str | None:
    if label is None:
        return None
    text = str(label).strip().lower()
    if not text:
        return None
    return OLD_SEGMENT_LABEL_MAP.get(text, text)


def normalize_phase_label(label: Any) -> str | None:
    if label is None:
        return None
    text = str(label).strip().lower()
    if not text:
        return None
    return OLD_PHASE_LABEL_MAP.get(text, text)


def normalize_activity_label(label: Any) -> str | None:
    if label is None:
        return None
    text = str(label).strip()
    if not text:
        return None
    mapped = OLD_ACTIVITY_LABEL_MAP.get(text.lower(), text)
    return mapped


def normalize_randori_kind(kind: Any) -> str | None:
    if kind is None:
        return None
    text = str(kind).strip().lower()
    if not text:
        return None
    return OLD_RANDORI_KIND_MAP.get(text, text)


def migrate_segment_types(segments: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not segments:
        return None
    migrated: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        item = dict(segment)
        original_label = item.get("label", item.get("type"))
        migrated_label = normalize_segment_label(original_label) or "technique"
        if "label" in item or "type" not in item:
            item["label"] = migrated_label
        if "type" in item:
            item["type"] = migrated_label
        migrated.append(item)
    return migrated or None


def migrate_processed_session_dict(data: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    migrated = dict(data)
    changed = False

    previous_version = migrated.get("segment_type_version")
    try:
        parsed_version = int(previous_version) if previous_version is not None else None
    except (TypeError, ValueError):
        parsed_version = None
    needs_v2_migration = parsed_version is None or parsed_version < SEGMENT_TYPE_VERSION

    normalized_activity_label = normalize_activity_label(migrated.get("activity_label"))
    if normalized_activity_label != migrated.get("activity_label"):
        migrated["activity_label"] = normalized_activity_label
        changed = True

    if needs_v2_migration:
        migrated_segments = migrate_segment_types(migrated.get("fc_phase_segments"))
        if migrated_segments != migrated.get("fc_phase_segments"):
            migrated["fc_phase_segments"] = migrated_segments
            changed = True

        phases = migrated.get("judo_phases")
        if phases:
            next_phases: list[dict[str, Any]] = []
            for phase in phases:
                if not isinstance(phase, dict):
                    continue
                item = dict(phase)
                new_label = normalize_phase_label(item.get("phase_label") or item.get("label"))
                if new_label:
                    if item.get("phase_label") != new_label:
                        item["phase_label"] = new_label
                        changed = True
                    if item.get("label") is not None and item.get("label") != new_label:
                        item["label"] = new_label
                        changed = True
                    item["phase_category"] = new_label if new_label in {"randori_tw", "randori_nw"} else new_label
                    item["needs_randori_details"] = new_label in {"randori_tw", "randori_nw"}
                next_phases.append(item)
            migrated["judo_phases"] = next_phases or None

        blocks = migrated.get("judo_randori_blocks")
        if blocks:
            next_blocks: list[dict[str, Any]] = []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                item = dict(block)
                new_kind = normalize_randori_kind(item.get("randori_kind")) or "randori_tw"
                if item.get("randori_kind") != new_kind:
                    item["randori_kind"] = new_kind
                    changed = True
                entries = item.get("randori_entries") or []
                next_entries: list[dict[str, Any]] = []
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    next_entries.append(dict(entry))
                item["randori_entries"] = next_entries or item.get("randori_entries")
                next_blocks.append(item)
            migrated["judo_randori_blocks"] = next_blocks or None

        migrated["segment_type_version"] = SEGMENT_TYPE_VERSION
        changed = True
    else:
        migrated["segment_type_version"] = parsed_version

    if "description_synced_from_segmentation" not in migrated:
        migrated["description_synced_from_segmentation"] = False
        changed = True
    if "description_synced_at" not in migrated:
        migrated["description_synced_at"] = None
        changed = True
    if "rr_manual_annotations" not in migrated:
        migrated["rr_manual_annotations"] = None
        changed = True

    return migrated, changed


@dataclass(slots=True)
class PolarFileInfo:
    device_id: str
    session_id: str
    date: str
    heure: str
    file_type: str
    filepath: str


@dataclass(slots=True)
class ProcessedSession:
    session_id: str
    device_id: str
    date: str
    heure_debut: str
    fc_start_ts: str | None
    nb_battements_rr: int
    nb_points_hr: int
    duree_s: float
    bpm_moyen: float
    bpm_min: int
    bpm_max: int
    rr_min_ms: int | None
    rr_max_ms: int | None
    rr_filepath: str
    hr_filepath: str
    format_fichiers: str
    import_timestamp: str
    annotation: str | None = None
    nb_rr_removed_artifacts: int | None = None
    rr_cleaning_min_ms: int | None = None
    rr_cleaning_max_ms: int | None = None
    source_session_ids: list[str] | None = None
    segments_count: int | None = None
    gaps_count: int | None = None
    gap_durations_s: list[int] | None = None
    duplicates_removed: int | None = None
    incomplete_segments: int | None = None
    import_mode: str | None = None
    rr_clean_filepath: str | None = None
    fc_clean_filepath: str | None = None
    clean_export_timestamp: str | None = None
    activity_family: str | None = None
    activity_label: str | None = None
    is_activity_annotated: bool | None = None
    activity_annotated_at: str | None = None
    is_archived: bool | None = None
    archived_at: str | None = None
    updated_at: str | None = None
    judo_session_type: str | None = None
    judo_phases: list[dict] | None = None
    judo_randori_blocks: list[dict] | None = None
    activity_notes: str | None = None
    session_rpe: int | None = None
    fc_phase_segments: list[dict] | None = None
    is_temporally_annotated: bool | None = None
    temporally_annotated_at: str | None = None
    segment_type_version: int | None = SEGMENT_TYPE_VERSION
    description_synced_from_segmentation: bool | None = None
    description_synced_at: str | None = None
    rr_manual_annotations: list[dict] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessedSession":
        migrated_data, _ = migrate_processed_session_dict(data)
        allowed_fields = {field.name for field in fields(cls)}
        filtered_data = {key: value for key, value in migrated_data.items() if key in allowed_fields}
        return cls(**filtered_data)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "device_id": self.device_id,
            "date": self.date,
            "heure_debut": self.heure_debut,
            "annotation": self.annotation,
            "fc_start_ts": self.fc_start_ts,
            "nb_battements_rr": self.nb_battements_rr,
            "nb_points_hr": self.nb_points_hr,
            "duree_s": self.duree_s,
            "bpm_moyen": self.bpm_moyen,
            "bpm_min": self.bpm_min,
            "bpm_max": self.bpm_max,
            "rr_min_ms": self.rr_min_ms,
            "rr_max_ms": self.rr_max_ms,
            "rr_filepath": self.rr_filepath,
            "hr_filepath": self.hr_filepath,
            "format_fichiers": self.format_fichiers,
            "import_timestamp": self.import_timestamp,
            "nb_rr_removed_artifacts": self.nb_rr_removed_artifacts,
            "rr_cleaning_min_ms": self.rr_cleaning_min_ms,
            "rr_cleaning_max_ms": self.rr_cleaning_max_ms,
            "source_session_ids": self.source_session_ids,
            "segments_count": self.segments_count,
            "gaps_count": self.gaps_count,
            "gap_durations_s": self.gap_durations_s,
            "duplicates_removed": self.duplicates_removed,
            "incomplete_segments": self.incomplete_segments,
            "import_mode": self.import_mode,
            "rr_clean_filepath": self.rr_clean_filepath,
            "fc_clean_filepath": self.fc_clean_filepath,
            "clean_export_timestamp": self.clean_export_timestamp,
            "activity_family": self.activity_family,
            "activity_label": self.activity_label,
            "is_activity_annotated": self.is_activity_annotated,
            "activity_annotated_at": self.activity_annotated_at,
            "is_archived": self.is_archived,
            "archived_at": self.archived_at,
            "updated_at": self.updated_at,
            "judo_session_type": self.judo_session_type,
            "judo_phases": self.judo_phases,
            "judo_randori_blocks": self.judo_randori_blocks,
            "activity_notes": self.activity_notes,
            "session_rpe": self.session_rpe,
            "fc_phase_segments": self.fc_phase_segments,
            "is_temporally_annotated": self.is_temporally_annotated,
            "temporally_annotated_at": self.temporally_annotated_at,
            "segment_type_version": self.segment_type_version,
            "description_synced_from_segmentation": self.description_synced_from_segmentation,
            "description_synced_at": self.description_synced_at,
            "rr_manual_annotations": self.rr_manual_annotations,
        }
