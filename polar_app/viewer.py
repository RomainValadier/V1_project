from __future__ import annotations

import pandas as pd

from polar_app.models import ProcessedSession
from polar_app.repository import ProcessedSessionRepository


class PolarSessionViewer:
    def __init__(self, repository: ProcessedSessionRepository) -> None:
        self.repository = repository

    def choose_sessions_interactively(self) -> list[str]:
        sessions = self.repository.list_sessions()

        if not sessions:
            print("Aucune seance traitee disponible dans processed/.")
            return []

        print("\nSeances disponibles :")
        for index, session in enumerate(sessions, start=1):
            print(
                f"  [{index}] {session.date} | {session.annotation or session.session_id}"
                f" | {session.heure_debut} | FC moy={session.bpm_moyen} | duree={session.duree_s:.1f}s"
            )

        user_input = input(
            "\nChoisis une ou plusieurs seances (ex: 1,3) ou 'all' pour tout afficher : "
        ).strip()

        if not user_input:
            return []
        if user_input.lower() == "all":
            return [session.session_id for session in sessions]

        selected_indexes: list[int] = []
        for chunk in user_input.split(","):
            value = chunk.strip()
            if not value:
                continue
            if not value.isdigit():
                raise ValueError(f"Selection invalide : {value}")

            selected_indexes.append(int(value))

        invalid_indexes = [index for index in selected_indexes if index < 1 or index > len(sessions)]
        if invalid_indexes:
            raise ValueError(
                "Index de seance invalide(s) : " + ", ".join(str(index) for index in invalid_indexes)
            )

        return [sessions[index - 1].session_id for index in selected_indexes]

    def display_sessions(self, session_ids: list[str] | None = None, head: int = 10) -> None:
        sessions = self.repository.get_sessions(session_ids)

        if not sessions:
            print("Aucune seance traitee disponible dans processed/.")
            return

        print(f"\n{'=' * 60}")
        print("  Affichage des donnees Polar (RR / FC)")
        print(f"{'=' * 60}")

        for session in sessions:
            _, rr_frame, hr_frame = self.repository.load_session_data(session.session_id)
            self._display_session_summary(session)
            self._display_frame_preview("Apercu RR", rr_frame, head)
            self._display_frame_preview("Apercu FC", hr_frame, head)

    def _display_session_summary(self, session: ProcessedSession) -> None:
        print(f"\nSeance : {session.annotation or session.session_id}")
        print(f"  Session ID   : {session.session_id}")
        print(f"  Date / heure : {session.date} {session.heure_debut}")
        print(f"  Duree        : {session.duree_s:.1f} s")
        print(f"  RR           : {session.nb_battements_rr} battements")
        print(
            f"  FC           : {session.nb_points_hr} points | "
            f"min={session.bpm_min} max={session.bpm_max} moyen={session.bpm_moyen}"
        )

    def _display_frame_preview(self, title: str, frame: pd.DataFrame, head: int) -> None:
        print(f"\n  {title}")
        print(f"  {'-' * len(title)}")

        if frame.empty:
            print("  (aucune donnee)")
            return

        preview = frame.head(head).to_string(index=False)
        for line in preview.splitlines():
            print(f"  {line}")
