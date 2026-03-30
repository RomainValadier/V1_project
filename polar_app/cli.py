from __future__ import annotations

import argparse

from polar_app.importer import PolarImporter
from polar_app.repository import ProcessedSessionRepository
from polar_app.viewer import PolarSessionViewer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import et visualisation des fichiers Polar H10")
    parser.add_argument(
        "--source",
        default="C:/5eme/stage/V1_project/Polar_files",
        help="Dossier contenant les fichiers Polar .txt",
    )
    parser.add_argument(
        "--output",
        default="C:/5eme/stage/V1_project/data",
        help="Dossier de sortie des donnees",
    )
    parser.add_argument(
        "--show-session",
        nargs="+",
        help="Session_id(s) a afficher depuis data/processed",
    )
    parser.add_argument(
        "--show-all-sessions",
        action="store_true",
        help="Affiche toutes les seances deja traitees",
    )
    parser.add_argument(
        "--interactive-show",
        action="store_true",
        help="Propose une selection interactive des seances a afficher",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=10,
        help="Nombre de lignes a afficher par tableau RR/FC",
    )
    parser.add_argument(
        "--streamlit-help",
        action="store_true",
        help="Affiche la commande a lancer pour l'interface Streamlit",
    )
    return parser


def run_cli() -> None:
    args = build_parser().parse_args()

    repository = ProcessedSessionRepository(args.output)
    viewer = PolarSessionViewer(repository)

    if args.interactive_show:
        selected_ids = viewer.choose_sessions_interactively()
        if selected_ids:
            viewer.display_sessions(selected_ids, head=args.head)
        return

    if args.show_all_sessions:
        viewer.display_sessions(head=args.head)
        return

    if args.show_session:
        viewer.display_sessions(args.show_session, head=args.head)
        return

    if args.streamlit_help:
        print("Lance l'interface avec : streamlit run streamlit_app.py")
        return

    importer = PolarImporter(args.source, args.output)
    importer.import_sessions()
