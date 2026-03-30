from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from polar_app.models import ProcessedSession


class ProcessedSessionRepository:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.processed_root = os.path.join(output_dir, "processed")
        self.clean_root = os.path.join(output_dir, "clean")

    def list_sessions(self) -> list[ProcessedSession]:
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
            sessions.append(session)

        return sessions

    def get_sessions(self, session_ids: list[str] | None = None) -> list[ProcessedSession]:
        sessions = self.list_sessions()
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
        session = self.get_sessions([session_id])[0]
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
        session = self.get_sessions([session_id])[0]
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
        session = self.get_sessions([session_id])[0]
        meta_path = os.path.join(self.processed_root, session.session_id, "session_meta.json")
        with open(meta_path, encoding="utf-8-sig") as file_obj:
            meta = json.load(file_obj)
        meta["annotation"] = annotation.strip() or session.annotation or session.session_id
        with open(meta_path, "w", encoding="utf-8") as file_obj:
            json.dump(meta, file_obj, ensure_ascii=False, indent=2)

    def delete_session(self, session_id: str) -> None:
        for branch in ("raw", "processed"):
            target_dir = os.path.join(self.output_dir, branch, session_id)
            if os.path.isdir(target_dir):
                shutil.rmtree(target_dir)

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
