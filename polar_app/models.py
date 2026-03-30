from __future__ import annotations

from dataclasses import dataclass, fields


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

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessedSession":
        allowed_fields = {field.name for field in fields(cls)}
        filtered_data = {key: value for key, value in data.items() if key in allowed_fields}
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
        }
