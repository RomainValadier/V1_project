from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from polar_app.analytics import build_hr_chart_frame, build_rr_chart_frame, compute_hr_missing_stats, compute_rr_missing_stats
from polar_app.importer import PolarImporter
from polar_app.repository import ProcessedSessionRepository

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(218, 165, 32, 0.10), transparent 30%),
                radial-gradient(circle at top right, rgba(46, 139, 87, 0.10), transparent 28%),
                linear-gradient(180deg, #f6f3ea 0%, #fbfaf7 52%, #f3f6f1 100%);
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }
        h1, h2, h3 {
            letter-spacing: -0.02em;
            color: #1d2a22;
        }
        .hero-card {
            padding: 1rem 1.2rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(246,248,243,0.94));
            border: 1px solid rgba(46, 84, 61, 0.10);
            box-shadow: 0 12px 36px rgba(41, 60, 47, 0.08);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 1.85rem;
            font-weight: 700;
            color: #163325;
            margin-bottom: 0.2rem;
        }
        .hero-subtitle {
            color: #526357;
            font-size: 0.98rem;
        }
        .section-chip {
            display: inline-block;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            background: rgba(55, 110, 79, 0.10);
            color: #2d5f43;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .mini-card {
            background: rgba(255,255,255,0.86);
            border: 1px solid rgba(33, 56, 42, 0.08);
            border-radius: 14px;
            padding: 0.7rem 0.85rem;
            margin-bottom: 0.55rem;
            box-shadow: 0 8px 24px rgba(41, 60, 47, 0.05);
        }
        .mini-label {
            color: #5e6c63;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.15rem;
        }
        .mini-value {
            color: #1b3024;
            font-size: 1.05rem;
            font-weight: 700;
            line-height: 1.1;
        }
        .mini-note {
            color: #66756c;
            font-size: 0.78rem;
            line-height: 1.3;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.84);
            border: 1px solid rgba(28, 51, 38, 0.08);
            padding: 0.65rem 0.8rem;
            border-radius: 14px;
            box-shadow: 0 6px 18px rgba(41, 60, 47, 0.05);
        }
        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f7f3e7 0%, #f5f7f2 100%);
            border-right: 1px solid rgba(42, 67, 51, 0.10);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Polar Sessions Dashboard</div>
            <div class="hero-subtitle">Visualisation claire des donnees RR brutes et FC brutes, seance par seance.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_label(label: str) -> None:
    st.markdown(f'<div class="section-chip">{label}</div>', unsafe_allow_html=True)


def render_mini_card(label: str, value: str, note: str | None = None) -> None:
    note_html = f'<div class="mini-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">{label}</div>
            <div class="mini-value">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_duration(duration_seconds: float | int | None) -> str:
    if duration_seconds is None:
        return "-"

    total_minutes = int(round(float(duration_seconds) / 60))
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours} heure {minutes} minutes"


def parse_session_start(session) -> datetime | None:
    if session.fc_start_ts:
        return datetime.fromisoformat(session.fc_start_ts)
    return datetime.fromisoformat(f"{session.date}T{session.heure_debut}")


def get_session_end(session) -> datetime | None:
    start_dt = parse_session_start(session)
    if start_dt is None or session.duree_s is None:
        return None
    return start_dt + timedelta(seconds=float(session.duree_s))


def format_datetime_label(dt: datetime | None) -> str:
    if dt is None:
        return "-"
    return dt.strftime("%H:%M:%S")


def compute_sampling_frequency(frame: pd.DataFrame) -> float | None:
    if frame.empty or "t_offset_ms" not in frame.columns:
        return None

    duration_s = float(frame["t_offset_ms"].max()) / 1000
    if duration_s <= 0:
        return None

    return len(frame) / duration_s


def format_frequency(freq_hz: float | None) -> str:
    if freq_hz is None:
        return "-"
    return f"{freq_hz:.2f} Hz"


def format_optional_int(value) -> str:
    return "-" if value is None else str(value)


def build_chart_series(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    if frame.empty or value_column not in frame.columns:
        return pd.DataFrame()
    columns = ["t_min", value_column]
    if "line_group" in frame.columns:
        columns.append("line_group")
    return frame[columns].copy()



def chart_has_values(frame: pd.DataFrame, value_column: str) -> bool:
    return not frame.empty and value_column in frame.columns and frame[value_column].notna().any()



def render_time_series_chart(frame: pd.DataFrame, value_column: str, y_title: str, height: int = 260) -> None:
    if not chart_has_values(frame, value_column):
        return

    encode_kwargs = {
        "x": alt.X("t_min:Q", title="Temps (min)"),
        "y": alt.Y(f"{value_column}:Q", title=y_title),
    }
    if "line_group" in frame.columns:
        encode_kwargs["detail"] = "line_group:N"

    chart = alt.Chart(frame.dropna(subset=[value_column])).mark_line(strokeWidth=2).encode(**encode_kwargs)
    st.altair_chart(chart.properties(height=height), use_container_width=True)


def render_dashboard_notice() -> None:
    notice = st.session_state.pop("dashboard_notice", None)
    if not notice:
        return
    level = notice.get("level", "info")
    message = notice.get("message", "")
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)


def format_session_label(session) -> str:
    annotation = session.annotation or session.session_id
    return (
        f"{annotation} | {session.heure_debut} | "
        f"FC moy {session.bpm_moyen} bpm | duree {format_duration(session.duree_s)}"
    )


def sort_sessions_by_day_and_time(sessions):
    return sorted(sessions, key=lambda session: (session.date, session.heure_debut, session.session_id))


def detect_uploaded_sessions(uploaded_files):
    importer = PolarImporter("", DEFAULT_OUTPUT_DIR)
    detected = {}

    for uploaded_file in uploaded_files:
        file_info = importer.parse_filename(uploaded_file.name)
        if file_info is None:
            continue

        if file_info.session_id not in detected:
            detected[file_info.session_id] = {
                "session_id": file_info.session_id,
                "date": file_info.date,
                "heure_debut": file_info.heure,
                "types": set(),
            }
        detected[file_info.session_id]["types"].add(file_info.file_type)

    return detected


def render_import_section(output_dir: str) -> bool:
    render_section_label("Import")
    uploaded_files = st.file_uploader(
        "Ajoute ici les fichiers Polar HR et RR (.txt)",
        type=["txt"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        return False

    detected_sessions = detect_uploaded_sessions(uploaded_files)
    if not detected_sessions:
        st.error("Aucun fichier Polar valide detecte dans la selection.")
        return False

    st.caption("Les fichiers sont groupes par session a partir de leur nom Polar.")

    annotation_overrides = {}
    for session_id, info in sorted(detected_sessions.items()):
        types = ", ".join(sorted(info["types"]))
        st.markdown(f"**{session_id}** - {info['date']} {info['heure_debut']} - types: {types}")
        annotation_overrides[session_id] = st.text_input(
            f"Nom de l'activite pour {session_id}",
            value=f"Activite {info['heure_debut']}",
            key=f"annotation_{session_id}",
        )

    invalid_sessions = [
        session_id for session_id, info in detected_sessions.items() if info["types"] != {"HR", "RR"}
    ]
    if invalid_sessions:
        st.warning(
            "Certaines sessions sont incompletes. Il faut un fichier HR et un fichier RR pour chacune : "
            + ", ".join(invalid_sessions)
        )

    if not st.button("Importer les fichiers Polar", type="primary", use_container_width=True):
        return False

    if invalid_sessions:
        st.error("Import annule tant que toutes les sessions uploadees ne sont pas completes.")
        return False

    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            destination = os.path.join(temp_dir, uploaded_file.name)
            with open(destination, "wb") as file_obj:
                file_obj.write(uploaded_file.getbuffer())

        importer = PolarImporter(temp_dir, output_dir)
        imported_sessions = importer.import_sessions(
            annotation_overrides=annotation_overrides,
            prompt_for_annotation=False,
        )

    if imported_sessions:
        st.success("Import termine pour : " + ", ".join(imported_sessions))
        return True

    st.error("Aucune session n'a pu etre importee.")
    return False


def render_session_metrics(session, rr_frame: pd.DataFrame, hr_frame: pd.DataFrame) -> None:
    start_dt = parse_session_start(session)
    end_dt = get_session_end(session)
    rr_frequency = compute_sampling_frequency(rr_frame)
    hr_frequency = compute_sampling_frequency(hr_frame)
    rr_missing_points, rr_missing_pct = compute_rr_missing_stats(rr_frame)
    hr_missing_points, hr_missing_pct = compute_hr_missing_stats(hr_frame)

    overview_columns = st.columns(5)
    overview_columns[0].metric("Date", session.date)
    overview_columns[1].metric("Debut", format_datetime_label(start_dt))
    overview_columns[2].metric("Fin", format_datetime_label(end_dt))
    overview_columns[3].metric("Duree", format_duration(session.duree_s))
    overview_columns[4].metric("Appareil", session.device_id)

    rr_columns = st.columns(5)
    rr_columns[0].metric("Points RR", f"{len(rr_frame)}")
    rr_columns[1].metric("Freq RR", format_frequency(rr_frequency))
    rr_columns[2].metric("RR min", f"{format_optional_int(session.rr_min_ms)} ms")
    rr_columns[3].metric("RR max", f"{format_optional_int(session.rr_max_ms)} ms")
    rr_columns[4].metric("RR manquants", f"{rr_missing_pct:.1f} %", delta=f"{rr_missing_points} points")

    fc_columns = st.columns(6)
    fc_columns[0].metric("Points FC", f"{len(hr_frame)}")
    fc_columns[1].metric("Freq FC", format_frequency(hr_frequency))
    fc_columns[2].metric("FC moy", f"{session.bpm_moyen} bpm")
    fc_columns[3].metric("FC min", f"{session.bpm_min} bpm")
    fc_columns[4].metric("FC max", f"{session.bpm_max} bpm")
    fc_columns[5].metric("FC manquants", f"{hr_missing_pct:.1f} %", delta=f"{hr_missing_points} points")


def render_session_management(repository: ProcessedSessionRepository, session) -> None:
    render_section_label("Gestion")
    rename_key = f"rename_input_{session.session_id}"
    if rename_key not in st.session_state:
        st.session_state[rename_key] = session.annotation or session.session_id

    st.text_input(
        "Nom de l'activite",
        key=rename_key,
        help="Modifie l'annotation affichee dans le dashboard et conservee dans les metadonnees.",
    )

    action_cols = st.columns(2)
    if action_cols[0].button("Enregistrer le nouveau nom", key=f"rename_button_{session.session_id}", use_container_width=True):
        repository.update_annotation(session.session_id, st.session_state.get(rename_key, session.annotation or session.session_id))
        st.session_state["dashboard_notice"] = {
            "level": "success",
            "message": f"Nom mis a jour pour {session.session_id}.",
        }
        st.rerun()

    confirm_key = f"delete_confirm_{session.session_id}"
    action_cols[1].checkbox("Confirmer la suppression", key=confirm_key)
    if st.button("Supprimer cette activite", key=f"delete_button_{session.session_id}", use_container_width=True):
        if not st.session_state.get(confirm_key, False):
            st.warning("Coche la confirmation avant de supprimer cette activite.")
        else:
            repository.delete_session(session.session_id)
            st.session_state["dashboard_notice"] = {
                "level": "success",
                "message": f"Activite supprimee : {session.annotation or session.session_id}.",
            }
            st.rerun()


def render_session_charts(repository: ProcessedSessionRepository, session, rr_frame: pd.DataFrame, hr_frame: pd.DataFrame, show_tables: bool) -> None:
    rr_raw_chart = build_rr_chart_frame(rr_frame)
    hr_chart = build_hr_chart_frame(hr_frame)
    rr_series = build_chart_series(rr_raw_chart, "rr_interval_ms")
    hr_series = build_chart_series(hr_chart, "bpm")
    end_dt = get_session_end(session)

    session_card = st.container()
    with session_card:
        render_section_label("Seance")
        st.subheader(f"{session.annotation or session.session_id}")
        st.caption(
            f"Session ID : {session.session_id} | {session.date} | debut {format_datetime_label(parse_session_start(session))} | "
            f"fin {format_datetime_label(end_dt)} | duree {format_duration(session.duree_s)} | format {session.format_fichiers} | "
            f"import {session.import_timestamp}"
        )
        render_session_metrics(session, rr_frame, hr_frame)
        render_session_management(repository, session)

        render_section_label("RR brut")
        if not chart_has_values(rr_series, "rr_interval_ms"):
            st.info("Aucune donnee RR disponible pour cette seance.")
        else:
            render_time_series_chart(rr_series, "rr_interval_ms", "RR (ms)", height=260)
        st.caption("Intervalles RR bruts importes depuis le fichier Polar.")

        render_section_label("FC brute")
        if not chart_has_values(hr_series, "bpm"):
            st.info("Aucune donnee FC disponible pour cette seance.")
        else:
            render_time_series_chart(hr_series, "bpm", "FC (bpm)", height=260)
        st.caption("Frequence cardiaque brute en bpm.")


        with st.expander("Apercu des donnees brutes"):
            st.markdown("**RR brut**")
            st.dataframe(rr_frame.head(20), use_container_width=True)
            st.markdown("**FC brute**")
            st.dataframe(hr_frame.head(20), use_container_width=True)

        if show_tables:
            st.markdown("**Donnees completes**")
            st.markdown("**RR brut**")
            st.dataframe(rr_frame, use_container_width=True, height=240)
            st.markdown("**FC brute**")
            st.dataframe(hr_frame, use_container_width=True, height=240)

        st.divider()


def main() -> None:
    st.set_page_config(
        page_title="Polar Sessions Dashboard",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_styles()
    render_page_header()
    render_dashboard_notice()

    with st.sidebar:
        output_dir = st.text_input("Dossier data", value=DEFAULT_OUTPUT_DIR)
        st.caption("L'import se fait desormais depuis la page Import Polar.")

        render_section_label("Affichage")
        repository = ProcessedSessionRepository(output_dir)
        sessions = sort_sessions_by_day_and_time(repository.list_sessions())

        if not sessions:
            st.warning("Aucune seance traitee trouvee dans ce dossier.")
            st.stop()

        available_days = sorted({session.date for session in sessions})
        selected_day = st.selectbox("Jour", options=available_days, index=len(available_days) - 1)

        day_sessions = [session for session in sessions if session.date == selected_day]
        session_options = {format_session_label(session): session.session_id for session in day_sessions}
        default_labels = [list(session_options.keys())[-1]] if session_options else []
        selected_labels = st.multiselect(
            "Activites",
            options=list(session_options.keys()),
            default=default_labels,
        )
        show_tables = st.toggle("Afficher les tableaux complets", value=True)

    if not selected_labels:
        st.info("Choisis au moins une seance dans la barre laterale.")
        st.stop()

    selected_ids = [session_options[label] for label in selected_labels]
    selected_sessions = repository.get_sessions(selected_ids)

    for session in selected_sessions:
        _, rr_frame, hr_frame = repository.load_session_data(session.session_id)
        render_session_charts(repository, session, rr_frame, hr_frame, show_tables)


if __name__ == "__main__":
    main()

