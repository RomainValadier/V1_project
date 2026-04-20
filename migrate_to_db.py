#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "schema_v1_1.sql"
DEFAULT_DB_PATH = PROJECT_ROOT / "projet_i.db"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

LOCAL_TZ = ZoneInfo("Europe/Paris")
ATHLETE_ID = "romain_valadier"
UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "projet-i-insep-sqlite-v1.1")

QUALITY_LABEL_MAP = {
    "OK": "OK",
    "WARNING": "WARNING",
    "WARN": "WARNING",
    "ALERTE": "WARNING",
    "BAD": "BAD",
    "EXCLUSION": "BAD",
}

TYPE_SEANCE_VALUES = {"randori_tw", "randori_nw", "randori_mixte", "technique", "autre"}
PHASE_TYPE_VALUES = {
    "randori_tw",
    "randori_nw",
    "randori_mixte",
    "recuperation",
    "echauffement",
    "technique",
    "retour_calme",
    "autre",
}
RANDORI_TYPES = {"randori_tw", "randori_nw", "randori_mixte"}
ALIAS_MAP = {
    "randori": "randori_tw",
    "randoris": "randori_tw",
    "randoris_tw": "randori_tw",
    "randori_tw": "randori_tw",
    "tw": "randori_tw",
    "randoris_nw": "randori_nw",
    "randori_nw": "randori_nw",
    "nw": "randori_nw",
    "nw_pure": "randori_nw",
    "randoris_nw_pure": "randori_nw",
    "randori_mixte": "randori_mixte",
    "randoris_mixte": "randori_mixte",
    "mixtes_nw_tw": "randori_mixte",
    "technique": "technique",
    "echauffement": "echauffement",
    "recuperation": "recuperation",
    "retour_calme": "retour_calme",
    "retour_calme_": "retour_calme",
    "prepa": "autre",
    "musculation": "autre",
    "cardio_aerobie": "autre",
    "cardio_fractionnee": "autre",
    "autres": "autre",
    "autre": "autre",
}


def stable_id(kind: str, *parts: Any) -> str:
    payload = "|".join([kind, *(str(part) for part in parts)])
    return str(uuid.uuid5(UUID_NAMESPACE, payload))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_label(value: Any) -> str:
    if value is None:
        return "autre"
    text = str(value).strip()
    if not text:
        return "autre"
    last_part = [part.strip() for part in text.split(">") if part.strip()]
    text = last_part[-1] if last_part else text
    text = text.lower()
    text = text.replace("é", "e").replace("è", "e").replace("ê", "e")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    normalized = ALIAS_MAP.get(text, "autre")
    if normalized == "autre" and text not in ALIAS_MAP:
        logging.warning("Label inconnu '%s' -> autre", value)
    return normalized


def normalize_type_seance(meta: dict[str, Any]) -> str:
    label = normalize_label(meta.get("activity_label") or meta.get("judo_session_type"))
    return label if label in TYPE_SEANCE_VALUES else "autre"


def normalize_quality_label(value: Any) -> str | None:
    if value is None:
        return None
    return QUALITY_LABEL_MAP.get(str(value).strip().upper(), "BAD")


def normalize_pipeline_version(value: Any) -> str:
    raw = str(value or "3.3").strip()
    return raw if raw.startswith("v") else f"v{raw}"


def parse_local_datetime(value: Any) -> datetime | None:
    if value in (None, "", "NA"):
        return None
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TZ)
    return parsed


def session_start(meta: dict[str, Any]) -> datetime:
    candidate = parse_local_datetime(meta.get("fc_start_ts"))
    if candidate is not None:
        return candidate

    date_value = meta.get("date")
    if not date_value:
        session_id = str(meta["session_id"])
        date_value = f"{session_id[:4]}-{session_id[4:6]}-{session_id[6:8]}"
    time_value = meta.get("heure_debut") or "00:00:00"
    candidate = parse_local_datetime(f"{date_value}T{time_value}")
    if candidate is None:
        raise ValueError(f"Impossible de determiner le debut de session {meta.get('session_id')}")
    return candidate


def to_utc_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def to_utc_ms(value: datetime) -> int:
    return int(round(value.astimezone(timezone.utc).timestamp() * 1000))


def optional_int(value: Any) -> int | None:
    if value in (None, "", "NA"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def rel(path: str | Path | None, root: Path) -> str | None:
    if path in (None, ""):
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        return str(candidate.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def rel_data_path(path: str | Path | None, data_root: Path, project_root: Path) -> str | None:
    if path in (None, ""):
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = data_root / candidate
    return rel(candidate, project_root)


def get_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row
    return conn


def initialize_schema(conn: sqlite3.Connection, schema_path: Path) -> None:
    conn.executescript(schema_path.read_text(encoding="utf-8"))


def repair_derived_timing_schema(conn: sqlite3.Connection, schema_path: Path) -> None:
    phases_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='phases_realisees'"
    ).fetchone()
    if not phases_sql or "phases_programmees_old" not in str(phases_sql["sql"]):
        return

    logging.warning(
        "Schema phases_realisees repare : reconstruction des tables derivees des annotations temporelles"
    )
    conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("DROP TABLE IF EXISTS rpe")
    conn.execute("DROP TABLE IF EXISTS randoris_realises")
    conn.execute("DROP TABLE IF EXISTS phases_realisees")
    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_schema(conn, schema_path)


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def ensure_schema_compatible(conn: sqlite3.Connection) -> None:
    phase_programmee_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='phases_programmees'"
    ).fetchone()
    if phase_programmee_sql and "retour_calme" not in str(phase_programmee_sql["sql"]):
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("ALTER TABLE phases_programmees RENAME TO phases_programmees_old")
        conn.execute(
            """
            CREATE TABLE phases_programmees (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                ordre INTEGER NOT NULL CHECK (ordre >= 1),
                type_phase TEXT NOT NULL CHECK (type_phase IN (
                    'randori_tw', 'randori_nw', 'randori_mixte', 'recuperation',
                    'echauffement', 'technique', 'retour_calme', 'autre'
                )),
                nb_repetitions INTEGER CHECK (nb_repetitions >= 1),
                duree_cible_s INTEGER CHECK (duree_cible_s > 0),
                recuperation_cible_s INTEGER CHECK (recuperation_cible_s >= 0),
                rpe_cible INTEGER CHECK (rpe_cible BETWEEN 0 AND 10),
                notes_entraineur TEXT,
                UNIQUE (session_id, ordre)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO phases_programmees (
                id, session_id, ordre, type_phase, nb_repetitions,
                duree_cible_s, recuperation_cible_s, rpe_cible, notes_entraineur
            )
            SELECT
                id, session_id, ordre,
                CASE WHEN type_phase = 'retour_calme' THEN 'retour_calme' ELSE type_phase END,
                nb_repetitions, duree_cible_s, recuperation_cible_s, rpe_cible, notes_entraineur
            FROM phases_programmees_old
            """
        )
        conn.execute("DROP TABLE phases_programmees_old")
        conn.execute("PRAGMA foreign_keys = ON")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS phases_realisees (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            phase_programmee_id TEXT REFERENCES phases_programmees(id) ON DELETE SET NULL,
            ordre INTEGER NOT NULL CHECK (ordre >= 1),
            type_phase TEXT NOT NULL CHECK (type_phase IN (
                'randori_tw', 'randori_nw', 'randori_mixte', 'recuperation',
                'echauffement', 'technique', 'retour_calme', 'autre'
            )),
            t_debut_utc_ms INTEGER NOT NULL,
            t_fin_utc_ms INTEGER NOT NULL,
            duree_reelle_s REAL GENERATED ALWAYS AS ((t_fin_utc_ms - t_debut_utc_ms) / 1000.0) STORED,
            source_annotation TEXT NOT NULL CHECK (
                source_annotation IN ('annotation_fc', 'observateur_manuel', 'video_ia')
            ),
            timing_annotation TEXT NOT NULL CHECK (timing_annotation IN ('in', 'post')),
            source_segment_index INTEGER,
            notes TEXT,
            CHECK (t_fin_utc_ms > t_debut_utc_ms),
            UNIQUE (session_id, ordre)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rr_manual_annotations (
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            source_index INTEGER NOT NULL CHECK (source_index >= 0),
            t_offset_ms INTEGER,
            rr_interval_ms REAL,
            manual_flag TEXT NOT NULL DEFAULT 'manuel',
            created_at TEXT,
            updated_at TEXT,
            PRIMARY KEY (session_id, source_index)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_phases_realisees_session ON phases_realisees (session_id, ordre)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_phases_realisees_programmee ON phases_realisees (phase_programmee_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rr_manual_annotations_session ON rr_manual_annotations (session_id, source_index)"
    )
    if "phase_realisee_id" not in table_columns(conn, "randoris_realises"):
        conn.execute(
            "ALTER TABLE randoris_realises ADD COLUMN phase_realisee_id TEXT REFERENCES phases_realisees(id) ON DELETE RESTRICT"
        )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_randoris_realises_phase_realisee ON randoris_realises (phase_realisee_id)"
    )

    sql_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='randoris_realises'"
    ).fetchone()
    if not sql_row:
        return
    sql = str(sql_row["sql"])
    if "annotation_fc" not in sql or "athlete_b_id IS NULL" not in sql:
        raise RuntimeError(
            "La table randoris_realises existe avec un schema incompatible. "
            "Utilise schema_v1_1.sql mis a jour ou recrée la BDD."
        )


def upsert_athlete(conn: sqlite3.Connection, dry_run: bool) -> None:
    if dry_run:
        return
    conn.execute(
        """
        INSERT INTO individus (id, nom, prenom, email, sexe, date_naissance, role, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1)
        ON CONFLICT(id) DO UPDATE SET
            nom=excluded.nom,
            prenom=excluded.prenom,
            email=excluded.email,
            sexe=excluded.sexe,
            date_naissance=excluded.date_naissance,
            role=excluded.role,
            is_active=excluded.is_active,
            updated_at=strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
        """,
        (
            ATHLETE_ID,
            "Valadier",
            "Romain",
            "romanopic@gmail.com",
            "M",
            "2002-07-20",
            "athlete",
        ),
    )
    conn.execute(
        """
        INSERT INTO athletes (individu_id, categorie_poids, taille_cm)
        VALUES (?, ?, ?)
        ON CONFLICT(individu_id) DO UPDATE SET
            categorie_poids=excluded.categorie_poids,
            taille_cm=excluded.taille_cm
        """,
        (ATHLETE_ID, "64 kg", 165.0),
    )


def upsert_pipeline_version(
    conn: sqlite3.Connection,
    version: str,
    date_deploy: str,
    params: dict[str, Any] | None,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    conn.execute(
        """
        INSERT INTO pipeline_versions (version, date_deploy, description, params_json, is_current)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(version) DO UPDATE SET
            params_json=COALESCE(pipeline_versions.params_json, excluded.params_json)
        """,
        (
            version,
            date_deploy,
            "Version reconstruite depuis clean_meta",
            json.dumps(params or {}, ensure_ascii=False, sort_keys=True),
            1 if version == "v3.3" else 0,
        ),
    )


def upsert_session(conn: sqlite3.Connection, meta: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    session_id = str(meta["session_id"])
    date_seance = meta.get("date") or f"{session_id[:4]}-{session_id[4:6]}-{session_id[6:8]}"
    conn.execute(
        """
        INSERT INTO sessions (
            id, date_seance, heure_debut_reelle, type_seance, activity_family,
            activity_notes, operateur, is_archived, archived_at,
            is_temporally_annotated, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            date_seance=excluded.date_seance,
            heure_debut_reelle=excluded.heure_debut_reelle,
            type_seance=excluded.type_seance,
            activity_family=excluded.activity_family,
            activity_notes=excluded.activity_notes,
            operateur=excluded.operateur,
            is_archived=excluded.is_archived,
            archived_at=excluded.archived_at,
            is_temporally_annotated=excluded.is_temporally_annotated,
            notes=excluded.notes
        """,
        (
            session_id,
            date_seance,
            meta.get("heure_debut"),
            normalize_type_seance(meta),
            meta.get("activity_family"),
            meta.get("activity_notes"),
            meta.get("operateur") or "Romain Valadier",
            1 if meta.get("is_archived") else 0,
            meta.get("archived_at"),
            1 if meta.get("is_temporally_annotated") else 0,
            meta.get("annotation"),
        ),
    )


def upsert_session_athlete(conn: sqlite3.Connection, meta: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    conn.execute(
        """
        INSERT INTO session_athletes (session_id, athlete_id, rpe_global_seance, rpe_global_timestamp)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(session_id, athlete_id) DO UPDATE SET
            rpe_global_seance=excluded.rpe_global_seance,
            rpe_global_timestamp=excluded.rpe_global_timestamp
        """,
        (
            meta["session_id"],
            ATHLETE_ID,
            optional_int(meta.get("session_rpe")),
            meta.get("updated_at") or meta.get("activity_annotated_at") or meta.get("import_timestamp"),
        ),
    )


def upsert_acquisition(
    conn: sqlite3.Connection,
    meta: dict[str, Any],
    pipeline_version: str,
    data_root: Path,
    project_root: Path,
    dry_run: bool,
) -> str:
    session_id = str(meta["session_id"])
    acquisition_id = stable_id("acquisition", session_id, ATHLETE_ID)
    if dry_run:
        return acquisition_id

    start_dt = session_start(meta)
    end_dt = start_dt + timedelta(seconds=float(meta.get("duree_s") or 0.0))
    conn.execute(
        """
        INSERT INTO acquisitions (
            id, session_id, athlete_id, pipeline_version, source, device_id,
            start_time_utc, end_time_utc, nb_rr_bruts, nb_points_hr, duree_s,
            segments_count, gaps_count, nb_deconnexions, rr_raw_filepath,
            hr_raw_filepath, statut, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id, athlete_id) DO UPDATE SET
            pipeline_version=excluded.pipeline_version,
            source=excluded.source,
            device_id=excluded.device_id,
            start_time_utc=excluded.start_time_utc,
            end_time_utc=excluded.end_time_utc,
            nb_rr_bruts=excluded.nb_rr_bruts,
            nb_points_hr=excluded.nb_points_hr,
            duree_s=excluded.duree_s,
            segments_count=excluded.segments_count,
            gaps_count=excluded.gaps_count,
            nb_deconnexions=excluded.nb_deconnexions,
            rr_raw_filepath=excluded.rr_raw_filepath,
            hr_raw_filepath=excluded.hr_raw_filepath,
            statut=excluded.statut,
            notes=excluded.notes
        """,
        (
            acquisition_id,
            session_id,
            ATHLETE_ID,
            pipeline_version,
            "psl",
            meta.get("device_id"),
            to_utc_z(start_dt),
            to_utc_z(end_dt),
            meta.get("nb_battements_rr"),
            meta.get("nb_points_hr"),
            meta.get("duree_s"),
            meta.get("segments_count"),
            meta.get("gaps_count"),
            meta.get("gaps_count"),
            rel_data_path(meta.get("rr_filepath"), data_root, project_root),
            rel_data_path(meta.get("hr_filepath"), data_root, project_root),
            meta.get("status") or "completed",
            meta.get("annotation"),
        ),
    )
    return acquisition_id


def import_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pandas est necessaire pour lire rr_clean.parquet et remplir "
            "rr_clean_exploitabilite. Installe les dependances du projet."
        ) from exc
    return pd


def parquet_exploitability_counts(rr_clean_path: Path) -> tuple[Any, ...]:
    pd = import_pandas()
    frame = pd.read_parquet(rr_clean_path)
    total = int(len(frame))

    def counts(column: str) -> tuple[int | None, int | None, float | None]:
        if column not in frame.columns:
            return None, None, None
        ok_total = int(frame[column].fillna(False).astype(bool).sum())
        non_viable_total = total - ok_total
        rate = round(non_viable_total / total, 6) if total > 0 else 0.0
        return ok_total, non_viable_total, rate

    fc_ok, fc_non, fc_rate = counts("fc_ok")
    hrr_ok, hrr_non, hrr_rate = counts("hrr_ok")
    rmssd_ok, rmssd_non, rmssd_rate = counts("rmssd_ok")
    return fc_ok, fc_non, fc_rate, hrr_ok, hrr_non, hrr_rate, rmssd_ok, rmssd_non, rmssd_rate


def upsert_rr_clean(
    conn: sqlite3.Connection,
    acquisition_id: str,
    session_id: str,
    clean_meta: dict[str, Any],
    pipeline_version: str,
    data_root: Path,
    project_root: Path,
    dry_run: bool,
) -> None:
    if dry_run:
        return

    conn.execute(
        """
        INSERT INTO rr_clean_meta (
            acquisition_id, export_timestamp, ok_rr_total, global_quality_label
        )
        VALUES (?, ?, ?, ?)
        ON CONFLICT(acquisition_id) DO UPDATE SET
            export_timestamp=excluded.export_timestamp,
            ok_rr_total=excluded.ok_rr_total,
            global_quality_label=excluded.global_quality_label
        """,
        (
            acquisition_id,
            clean_meta.get("export_timestamp"),
            clean_meta.get("ok_rr_total"),
            normalize_quality_label(clean_meta.get("global_quality_label")),
        ),
    )

    rr_clean_path = data_root / "clean" / session_id / "rr_clean.parquet"
    if rr_clean_path.exists():
        counts = parquet_exploitability_counts(rr_clean_path)
        conn.execute(
            """
            INSERT INTO rr_clean_exploitabilite (
                acquisition_id, fc_ok_total, fc_non_viable_total, fc_non_ok_rate,
                hrr_ok_total, hrr_non_viable_total, hrr_non_ok_rate,
                rmssd_ok_total, rmssd_non_viable_total, rmssd_non_ok_rate
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(acquisition_id) DO UPDATE SET
                fc_ok_total=excluded.fc_ok_total,
                fc_non_viable_total=excluded.fc_non_viable_total,
                fc_non_ok_rate=excluded.fc_non_ok_rate,
                hrr_ok_total=excluded.hrr_ok_total,
                hrr_non_viable_total=excluded.hrr_non_viable_total,
                hrr_non_ok_rate=excluded.hrr_non_ok_rate,
                rmssd_ok_total=excluded.rmssd_ok_total,
                rmssd_non_viable_total=excluded.rmssd_non_viable_total,
                rmssd_non_ok_rate=excluded.rmssd_non_ok_rate
            """,
            (acquisition_id, *counts),
        )

    export_timestamp = clean_meta.get("export_timestamp") or datetime.now(timezone.utc).isoformat()
    fc_params = {
        "fc_clean_window_seconds": clean_meta.get("fc_clean_window_seconds"),
        "fc_clean_window_mode": clean_meta.get("fc_clean_window_mode"),
        "fc_clean_min_viable_points": clean_meta.get("fc_clean_min_viable_points"),
    }
    indicator_rows = [
        ("rr_clean", data_root / "clean" / session_id / "rr_clean.parquet", None),
        ("fc_clean", data_root / "clean" / session_id / "fc_clean.parquet", fc_params),
    ]
    for indicator_type, filepath, params in indicator_rows:
        conn.execute(
            """
            INSERT INTO rr_indicator_files (
                id, acquisition_id, pipeline_version, indicator_type, filepath,
                params_json, computed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(acquisition_id, indicator_type) DO UPDATE SET
                pipeline_version=excluded.pipeline_version,
                filepath=excluded.filepath,
                params_json=excluded.params_json,
                computed_at=excluded.computed_at
            """,
            (
                stable_id("rr_indicator_file", acquisition_id, indicator_type),
                acquisition_id,
                pipeline_version,
                indicator_type,
                rel(filepath, project_root),
                json.dumps(params, ensure_ascii=False, sort_keys=True) if params else None,
                export_timestamp,
            ),
        )


def first_import_log_line(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                return line.strip()
    except OSError:
        return None
    return None


def upsert_import_log(
    conn: sqlite3.Connection,
    acquisition_id: str,
    meta: dict[str, Any],
    session_dir: Path,
    project_root: Path,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    log_path = session_dir / "import_log.txt"
    conn.execute(
        """
        INSERT INTO import_log (
            id, acquisition_id, fichier_source, type_fichier, date_import,
            nb_lignes, statut, message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            fichier_source=excluded.fichier_source,
            date_import=excluded.date_import,
            nb_lignes=excluded.nb_lignes,
            statut=excluded.statut,
            message=excluded.message
        """,
        (
            stable_id("import_log", acquisition_id, "psl_rr"),
            acquisition_id,
            rel(log_path, project_root),
            "psl_rr",
            meta.get("import_timestamp") or meta.get("fc_start_ts") or f"{meta.get('date')}T{meta.get('heure_debut')}",
            meta.get("nb_battements_rr"),
            "ok",
            first_import_log_line(log_path),
        ),
    )


def rebuild_rr_manual_annotations(conn: sqlite3.Connection, meta: dict[str, Any], dry_run: bool) -> int:
    session_id = str(meta["session_id"])
    annotations = [entry for entry in meta.get("rr_manual_annotations") or [] if isinstance(entry, dict)]
    if dry_run:
        return len(annotations)

    conn.execute("DELETE FROM rr_manual_annotations WHERE session_id=?", (session_id,))
    seen_indices: set[int] = set()
    inserted = 0
    for entry in annotations:
        source_index = optional_int(entry.get("source_index"))
        if source_index is None or source_index < 0 or source_index in seen_indices:
            continue
        seen_indices.add(source_index)
        conn.execute(
            """
            INSERT INTO rr_manual_annotations (
                session_id, source_index, t_offset_ms, rr_interval_ms,
                manual_flag, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                source_index,
                optional_int(entry.get("t_offset_ms")),
                float(entry["rr_interval_ms"]) if entry.get("rr_interval_ms") not in (None, "", "NA") else None,
                entry.get("manual_flag") or "manuel",
                entry.get("created_at"),
                entry.get("updated_at"),
            ),
        )
        inserted += 1
    return inserted


def randori_segments_from_fc(meta: dict[str, Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for segment in meta.get("fc_phase_segments") or []:
        if not isinstance(segment, dict):
            continue
        label = normalize_label(segment.get("label") or segment.get("type"))
        if label not in RANDORI_TYPES:
            continue
        try:
            start_offset_s = float(segment.get("start_offset_s"))
            end_offset_s = float(segment.get("end_offset_s"))
        except (TypeError, ValueError):
            continue
        if end_offset_s <= start_offset_s:
            continue
        segments.append(
            {
                "label": label,
                "start_offset_s": start_offset_s,
                "end_offset_s": end_offset_s,
                "source_index": segment.get("segment_index"),
            }
        )
    return segments


def phase_segments_from_fc(meta: dict[str, Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for index, segment in enumerate(meta.get("fc_phase_segments") or [], start=1):
        if not isinstance(segment, dict):
            continue
        label = normalize_label(segment.get("label") or segment.get("type"))
        type_phase = label if label in PHASE_TYPE_VALUES else "autre"
        try:
            start_offset_s = float(segment.get("start_offset_s"))
            end_offset_s = float(segment.get("end_offset_s"))
        except (TypeError, ValueError):
            continue
        if end_offset_s <= start_offset_s:
            continue
        source_index = optional_int(segment.get("segment_index"))
        ordre = int(source_index) + 1 if source_index is not None else index
        segments.append(
            {
                "ordre": ordre,
                "type_phase": type_phase,
                "start_offset_s": start_offset_s,
                "end_offset_s": end_offset_s,
                "source_segment_index": source_index,
                "phase_uid": segment.get("phase_uid"),
                "source": segment.get("source"),
                "locked": segment.get("locked"),
            }
        )
    return sorted(segments, key=lambda item: int(item["ordre"]))


def phase_programmee_records(meta: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, phase in enumerate(meta.get("judo_phases") or [], start=1):
        if not isinstance(phase, dict):
            continue
        type_phase = normalize_label(phase.get("phase_label") or phase.get("label"))
        if type_phase not in PHASE_TYPE_VALUES:
            type_phase = "autre"
        phase_index = optional_int(phase.get("phase_index"))
        ordre = phase_index + 1 if phase_index is not None else index
        phase_uid = phase.get("phase_uid")
        records.append(
            {
                "id": stable_id("phase_programmee", meta["session_id"], phase_uid or ordre),
                "ordre": ordre,
                "type_phase": type_phase,
                "phase_uid": phase_uid,
                "notes": json.dumps(phase, ensure_ascii=False, sort_keys=True),
            }
        )
    return sorted(records, key=lambda item: int(item["ordre"]))


def rebuild_phases_programmees(conn: sqlite3.Connection, meta: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    session_id = str(meta["session_id"])
    records = phase_programmee_records(meta)
    by_uid = {str(record["phase_uid"]): record["id"] for record in records if record.get("phase_uid")}
    by_type: dict[str, list[str]] = {}
    for record in records:
        by_type.setdefault(str(record["type_phase"]), []).append(str(record["id"]))
    unique_by_type = {type_phase: ids[0] for type_phase, ids in by_type.items() if len(ids) == 1}

    if dry_run:
        return {"by_uid": by_uid, "unique_by_type": unique_by_type}

    conn.execute("DELETE FROM phases_programmees WHERE session_id=?", (session_id,))
    for record in records:
        block = None
        for candidate in meta.get("judo_randori_blocks") or []:
            if isinstance(candidate, dict) and candidate.get("phase_uid") == record.get("phase_uid"):
                block = candidate
                break
        conn.execute(
            """
            INSERT INTO phases_programmees (
                id, session_id, ordre, type_phase, nb_repetitions,
                duree_cible_s, recuperation_cible_s, notes_entraineur
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                session_id,
                record["ordre"],
                record["type_phase"],
                optional_int(block.get("randori_count")) if isinstance(block, dict) else None,
                int(round(float(block.get("randori_duration_min")) * 60))
                if isinstance(block, dict) and block.get("randori_duration_min") not in (None, "", "NA")
                else None,
                int(round(float(block.get("rest_between_randoris_min")) * 60))
                if isinstance(block, dict) and block.get("rest_between_randoris_min") not in (None, "", "NA")
                else None,
                record["notes"],
            ),
        )
    return {"by_uid": by_uid, "unique_by_type": unique_by_type}


def rpe_entries_by_repetition(meta: dict[str, Any]) -> dict[int, dict[str, Any]]:
    entries: dict[int, dict[str, Any]] = {}
    for block in meta.get("judo_randori_blocks") or []:
        if not isinstance(block, dict):
            continue
        for entry in block.get("randori_entries") or []:
            if not isinstance(entry, dict):
                continue
            repetition = optional_int(entry.get("repetition_index"))
            rpe_value = optional_int(entry.get("rpe"))
            if repetition is None or rpe_value is None:
                continue
            entries[repetition] = entry
    return entries


def clear_realized_timing(conn: sqlite3.Connection, session_id: str, dry_run: bool) -> None:
    if dry_run:
        return
    conn.execute(
        "DELETE FROM rpe WHERE randori_id IN (SELECT id FROM randoris_realises WHERE session_id=?)",
        (session_id,),
    )
    conn.execute("DELETE FROM randoris_realises WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM phases_realisees WHERE session_id=?", (session_id,))


def rebuild_phases_realisees_randoris_and_rpe(
    conn: sqlite3.Connection,
    meta: dict[str, Any],
    phase_links: dict[str, Any],
    dry_run: bool,
) -> tuple[int, int, int]:
    session_id = str(meta["session_id"])
    phase_segments = phase_segments_from_fc(meta)
    rpe_entries = rpe_entries_by_repetition(meta)
    randori_segments = [segment for segment in phase_segments if segment["type_phase"] in RANDORI_TYPES]

    if rpe_entries and len(rpe_entries) != len(randori_segments):
        logging.warning(
            "[%s] %s RPE par randori pour %s segments FC randori : migration des correspondances fiables uniquement",
            session_id,
            len(rpe_entries),
            len(randori_segments),
        )

    if dry_run:
        return (
            len(phase_segments),
            len(randori_segments),
            sum(1 for index in range(1, len(randori_segments) + 1) if index in rpe_entries),
        )

    start_dt = session_start(meta)
    rpe_timestamp = (
        meta.get("updated_at")
        or meta.get("activity_annotated_at")
        or meta.get("import_timestamp")
        or to_utc_z(start_dt)
    )
    rpe_count = 0
    randori_number = 0

    for segment in phase_segments:
        hajime = start_dt + timedelta(seconds=segment["start_offset_s"])
        matte = start_dt + timedelta(seconds=segment["end_offset_s"])
        phase_uid = segment.get("phase_uid")
        phase_programmee_id = None
        if phase_uid and str(phase_uid) in phase_links["by_uid"]:
            phase_programmee_id = phase_links["by_uid"][str(phase_uid)]
        else:
            phase_programmee_id = phase_links["unique_by_type"].get(str(segment["type_phase"]))
        phase_realisee_id = stable_id("phase_realisee", session_id, segment["ordre"])
        notes = {
            "derived_from": "fc_phase_segments",
            "source": segment.get("source"),
            "locked": segment.get("locked"),
            "phase_uid": phase_uid,
        }
        conn.execute(
            """
            INSERT INTO phases_realisees (
                id, session_id, phase_programmee_id, ordre, type_phase,
                t_debut_utc_ms, t_fin_utc_ms, source_annotation,
                timing_annotation, source_segment_index, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, ordre) DO UPDATE SET
                phase_programmee_id=excluded.phase_programmee_id,
                type_phase=excluded.type_phase,
                t_debut_utc_ms=excluded.t_debut_utc_ms,
                t_fin_utc_ms=excluded.t_fin_utc_ms,
                source_annotation=excluded.source_annotation,
                timing_annotation=excluded.timing_annotation,
                source_segment_index=excluded.source_segment_index,
                notes=excluded.notes
            """,
            (
                phase_realisee_id,
                session_id,
                phase_programmee_id,
                segment["ordre"],
                segment["type_phase"],
                to_utc_ms(hajime),
                to_utc_ms(matte),
                "annotation_fc",
                "post",
                segment.get("source_segment_index"),
                json.dumps(notes, ensure_ascii=False, sort_keys=True),
            ),
        )

        if segment["type_phase"] not in RANDORI_TYPES:
            continue

        randori_number += 1
        randori_id = stable_id("randori_realise", session_id, randori_number)
        conn.execute(
            """
            INSERT INTO randoris_realises (
                id, session_id, phase_realisee_id, athlete_a_id, athlete_b_id, numero_randori,
                t_hajime_utc_ms, t_matte_utc_ms, type_randori,
                source_segmentation, confiance_segmentation, notes
            )
            VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, numero_randori) DO UPDATE SET
                phase_realisee_id=excluded.phase_realisee_id,
                athlete_a_id=excluded.athlete_a_id,
                athlete_b_id=NULL,
                t_hajime_utc_ms=excluded.t_hajime_utc_ms,
                t_matte_utc_ms=excluded.t_matte_utc_ms,
                type_randori=excluded.type_randori,
                source_segmentation=excluded.source_segmentation,
                confiance_segmentation=excluded.confiance_segmentation,
                notes=excluded.notes
            """,
            (
                randori_id,
                session_id,
                phase_realisee_id,
                ATHLETE_ID,
                randori_number,
                to_utc_ms(hajime),
                to_utc_ms(matte),
                segment["type_phase"],
                "annotation_fc",
                None,
                f"derived_from=phases_realisees; phase_realisee_id={phase_realisee_id}",
            ),
        )

        entry = rpe_entries.get(randori_number)
        if entry is None:
            continue
        rpe_value = optional_int(entry.get("rpe"))
        if rpe_value is None:
            continue
        conn.execute(
            """
            INSERT INTO rpe (
                id, randori_id, athlete_id, rpe_ressenti, moment_collecte, timestamp_collecte
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(randori_id, athlete_id) DO UPDATE SET
                rpe_ressenti=excluded.rpe_ressenti,
                moment_collecte=excluded.moment_collecte,
                timestamp_collecte=excluded.timestamp_collecte
            """,
            (
                stable_id("rpe", randori_id, ATHLETE_ID),
                randori_id,
                ATHLETE_ID,
                rpe_value,
                "post_randori",
                rpe_timestamp,
            ),
        )
        rpe_count += 1

    return len(phase_segments), randori_number, rpe_count


def migrate_session(
    conn: sqlite3.Connection,
    session_dir: Path,
    data_root: Path,
    project_root: Path,
    dry_run: bool,
) -> tuple[int, int, int, int]:
    meta_path = session_dir / "session_meta.json"
    meta = read_json(meta_path)
    session_id = str(meta["session_id"])
    clean_meta_path = data_root / "clean" / session_id / "clean_meta.json"
    clean_meta = read_json(clean_meta_path) if clean_meta_path.exists() else {}
    pipeline_version = normalize_pipeline_version(clean_meta.get("cleaning_algo_version") if clean_meta else "3.3")

    upsert_pipeline_version(
        conn,
        pipeline_version,
        str(meta.get("date") or session_id[:8]),
        clean_meta.get("params") if clean_meta else None,
        dry_run,
    )
    upsert_session(conn, meta, dry_run)
    upsert_session_athlete(conn, meta, dry_run)
    acquisition_id = upsert_acquisition(conn, meta, pipeline_version, data_root, project_root, dry_run)
    if clean_meta:
        upsert_rr_clean(conn, acquisition_id, session_id, clean_meta, pipeline_version, data_root, project_root, dry_run)
    upsert_import_log(conn, acquisition_id, meta, session_dir, project_root, dry_run)
    clear_realized_timing(conn, session_id, dry_run)
    phase_links = rebuild_phases_programmees(conn, meta, dry_run)
    phase_count, randori_count, rpe_count = rebuild_phases_realisees_randoris_and_rpe(conn, meta, phase_links, dry_run)
    rr_manual_count = rebuild_rr_manual_annotations(conn, meta, dry_run)
    return phase_count, randori_count, rpe_count, rr_manual_count


def log_orphan_clean_dirs(data_root: Path) -> int:
    processed_ids = {path.name for path in (data_root / "processed").iterdir() if path.is_dir()}
    orphan_count = 0
    clean_root = data_root / "clean"
    if not clean_root.exists():
        return 0
    for clean_dir in sorted(path for path in clean_root.iterdir() if path.is_dir()):
        if clean_dir.name not in processed_ids:
            logging.warning("Session orpheline clean ignoree : %s", clean_dir.name)
            orphan_count += 1
    return orphan_count


def table_count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migration PSL vers SQLite v1.1")
    parser.add_argument("--dry-run", action="store_true", help="Affiche ce qui serait migre sans ecrire")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Chemin de la BDD SQLite cible")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Chemin vers le dossier data")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH), help="Chemin vers schema_v1_1.sql")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data_root = Path(args.data_root).resolve()
    project_root = data_root.parent
    schema_path = Path(args.schema).resolve()
    db_path = Path(args.db).resolve()
    processed_root = data_root / "processed"

    if not processed_root.exists():
        raise FileNotFoundError(f"Dossier processed introuvable : {processed_root}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema introuvable : {schema_path}")

    conn = sqlite3.connect(":memory:") if args.dry_run else get_conn(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if args.dry_run:
        logging.info("DRY RUN : aucune ecriture persistante")

    initialize_schema(conn, schema_path)
    repair_derived_timing_schema(conn, schema_path)
    ensure_schema_compatible(conn)
    repair_derived_timing_schema(conn, schema_path)
    ensure_schema_compatible(conn)

    migrated = 0
    skipped = 0
    errors = 0
    phases_realisees = 0
    randoris = 0
    rpes = 0
    rr_manual_annotations = 0

    try:
        upsert_athlete(conn, args.dry_run)
        for session_dir in sorted(path for path in processed_root.iterdir() if path.is_dir()):
            if not (session_dir / "session_meta.json").exists():
                logging.warning("[%s] session_meta.json absent : ignoree", session_dir.name)
                skipped += 1
                continue
            try:
                session_phases, session_randoris, session_rpes, session_rr_manual = migrate_session(
                    conn, session_dir, data_root, project_root, args.dry_run
                )
                if not args.dry_run:
                    conn.commit()
                migrated += 1
                phases_realisees += session_phases
                randoris += session_randoris
                rpes += session_rpes
                rr_manual_annotations += session_rr_manual
                logging.info("[%s] migration OK", session_dir.name)
            except Exception as exc:
                if not args.dry_run:
                    conn.rollback()
                errors += 1
                logging.error("[%s] ERREUR : %s", session_dir.name, exc, exc_info=True)

        orphan_count = log_orphan_clean_dirs(data_root)
        if not args.dry_run:
            conn.commit()

        logging.info("Migration terminee")
        logging.info("Sessions migrees : %s", migrated)
        logging.info("Sessions ignorees : %s", skipped)
        logging.info("Sessions en erreur : %s", errors)
        logging.info("Phases realisees FC : %s", phases_realisees)
        logging.info("Randoris FC : %s", randoris)
        logging.info("RPE par randori : %s", rpes)
        logging.info("Annotations RR manuelles : %s", rr_manual_annotations)
        logging.info("Clean orphelins ignores : %s", orphan_count)

        if not args.dry_run:
            for table in (
                "sessions",
                "acquisitions",
                "session_athletes",
                "phases_programmees",
                "phases_realisees",
                "rr_clean_meta",
                "rr_clean_exploitabilite",
                "rr_indicator_files",
                "randoris_realises",
                "rpe",
                "rr_manual_annotations",
            ):
                logging.info("%s=%s", table, table_count(conn, table))
    finally:
        conn.close()

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
