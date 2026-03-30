from __future__ import annotations

import os
import tempfile
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from polar_app.analytics import build_hr_chart_frame, build_rr_chart_frame, compute_hr_missing_stats, compute_rr_missing_stats
from polar_app.importer import PolarImporter
from polar_app.repository import ProcessedSessionRepository

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


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
        .summary-card {
            background: linear-gradient(145deg, rgba(255,255,255,0.95), rgba(245,248,242,0.92));
            border: 1px solid rgba(33, 56, 42, 0.10);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 28px rgba(41, 60, 47, 0.07);
        }
        .summary-topline {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 1rem;
            margin-bottom: 0.45rem;
        }
        .summary-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #163325;
        }
        .summary-subtitle {
            color: #617268;
            font-size: 0.92rem;
        }
        .summary-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.55rem 0 0.8rem 0;
        }
        .summary-badge {
            display: inline-block;
            padding: 0.24rem 0.6rem;
            border-radius: 999px;
            background: rgba(55, 110, 79, 0.10);
            color: #2d5f43;
            font-size: 0.76rem;
            font-weight: 600;
        }
        .summary-note {
            color: #5e6c63;
            font-size: 0.84rem;
            line-height: 1.35;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_label(label: str) -> None:
    st.markdown(f'<div class="section-chip">{label}</div>', unsafe_allow_html=True)


def format_duration(duration_seconds: float | int | None) -> str:
    if duration_seconds is None:
        return "-"
    total_seconds = int(round(float(duration_seconds)))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min {seconds:02d} s"
    return f"{minutes} min {seconds:02d} s"


def format_gap_text(gaps: list[int] | None) -> str:
    if not gaps:
        return "Aucun gap"
    return " | ".join(f"{gap}s" for gap in gaps)


def format_datetime_label(value: datetime | None) -> str:
    return "-" if value is None else value.strftime("%Y-%m-%d %H:%M:%S")


def write_uploaded_files(uploaded_files, temp_dir: str) -> None:
    for uploaded_file in uploaded_files:
        destination = os.path.join(temp_dir, uploaded_file.name)
        with open(destination, "wb") as file_obj:
            file_obj.write(uploaded_file.getbuffer())


def build_plan(uploaded_files, output_dir: str, gap_merge_decisions: dict[str, bool] | None = None):
    with tempfile.TemporaryDirectory() as temp_dir:
        write_uploaded_files(uploaded_files, temp_dir)
        importer = PolarImporter(temp_dir, output_dir)
        return importer.plan_import(gap_merge_decisions=gap_merge_decisions)


def sort_sessions_by_time(repository: ProcessedSessionRepository, sessions):
    return sorted(
        sessions,
        key=lambda session: (
            repository.get_session_start(session) or datetime.min,
            session.session_id,
        ),
    )


def build_chart_series(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    if frame.empty or value_column not in frame.columns:
        return pd.DataFrame()
    columns = ["t_min", value_column]
    if "line_group" in frame.columns:
        columns.append("line_group")
    return frame[columns].copy()



def chart_has_values(frame: pd.DataFrame, value_column: str) -> bool:
    return not frame.empty and value_column in frame.columns and frame[value_column].notna().any()



def render_time_series_chart(frame: pd.DataFrame, value_column: str, y_title: str, height: int = 240) -> None:
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


def render_summary_card(title: str, subtitle: str, badges: list[str], note: str | None = None) -> None:
    badges_html = "".join(f'<span class="summary-badge">{badge}</span>' for badge in badges)
    note_html = f'<div class="summary-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-topline">
                <div class="summary-title">{title}</div>
                <div class="summary-subtitle">{subtitle}</div>
            </div>
            <div class="summary-badges">{badges_html}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_activity_preview(activity, activity_name: str) -> None:
    note = f"{len(activity.segments)} segment(s) source : " + ", ".join(segment.session_id for segment in activity.segments)
    render_summary_card(
        title=activity_name,
        subtitle=f"{activity.t_start.strftime('%Y-%m-%d')} | {activity.t_start.strftime('%H:%M:%S')} -> {activity.t_end.strftime('%H:%M:%S')}",
        badges=[
            f"Duree {format_duration(activity.duration_s)}",
            f"Segments {len(activity.segments)}",
            format_gap_text(activity.gaps_s),
            f"RR {len(activity.rr_frame)} points",
            f"FC {len(activity.hr_frame)} points",
        ],
        note=note,
    )

    metrics = st.columns(4)
    metrics[0].metric("Debut", activity.t_start.strftime("%H:%M:%S"))
    metrics[1].metric("Fin", activity.t_end.strftime("%H:%M:%S"))
    metrics[2].metric("Segments fusionnes", str(len(activity.segments)))
    metrics[3].metric("Deconnexions detectees", str(len(activity.gaps_s)))

    summary_rows = [
        {
            "segment_id": index,
            "session_id_source": segment.session_id,
            "debut": segment.t_start.strftime("%H:%M:%S"),
            "fin": segment.t_end.strftime("%H:%M:%S"),
            "duree": format_duration(segment.duration_s),
            "rr_points": segment.rr_count,
            "hr_points": segment.hr_count,
        }
        for index, segment in enumerate(activity.segments)
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    if activity.overlapping_sessions:
        overlaps = [
            f"{session.annotation or session.session_id} ({session.date} {session.heure_debut})"
            for session in activity.overlapping_sessions
        ]
        st.warning("Overlap detecte avec : " + " | ".join(overlaps))


def render_latest_raw_preview(activity) -> None:
    render_section_label("Donnees brutes detectees")
    st.caption("Apercu des donnees brutes pour l'activite la plus recente detectee.")
    rr_chart = build_rr_chart_frame(activity.rr_frame) if not activity.rr_frame.empty else pd.DataFrame()
    hr_chart = build_hr_chart_frame(activity.hr_frame) if not activity.hr_frame.empty else pd.DataFrame()

    rr_series = build_chart_series(rr_chart, "rr_interval_ms")
    hr_series = build_chart_series(hr_chart, "bpm")
    rr_missing_points, rr_missing_pct = compute_rr_missing_stats(activity.rr_frame)
    hr_missing_points, hr_missing_pct = compute_hr_missing_stats(activity.hr_frame)

    top_metrics = st.columns(2)
    top_metrics[0].metric("RR manquants", f"{rr_missing_pct:.1f} %", delta=f"{rr_missing_points} points")
    top_metrics[1].metric("FC manquants", f"{hr_missing_pct:.1f} %", delta=f"{hr_missing_points} points")

    if chart_has_values(rr_series, "rr_interval_ms"):
        render_time_series_chart(rr_series, "rr_interval_ms", "RR (ms)", height=240)
    else:
        st.info("Aucune valeur RR affichable sur cette activite.")

    if chart_has_values(hr_series, "bpm"):
        render_time_series_chart(hr_series, "bpm", "FC (bpm)", height=240)
    else:
        st.info("Aucune valeur FC affichable sur cette activite.")

    with st.expander("Apercu tabulaire"):
        st.markdown("**RR brut**")
        st.dataframe(activity.rr_frame.head(20), use_container_width=True)
        st.markdown("**FC brute**")
        st.dataframe(activity.hr_frame.head(20), use_container_width=True)


def render_imported_sessions(output_dir: str, imported_session_ids: list[str]) -> None:
    if not imported_session_ids:
        return

    repository = ProcessedSessionRepository(output_dir)
    sessions = sort_sessions_by_time(repository, repository.get_sessions(imported_session_ids))

    render_section_label("Resume import")
    for session in sessions:
        start_dt = repository.get_session_start(session)
        end_dt = repository.get_session_end(session)
        subtitle = f"{session.date} | {(start_dt.strftime('%H:%M:%S') if start_dt else '-')} -> {(end_dt.strftime('%H:%M:%S') if end_dt else '-')}"
        rr_frame, hr_frame = repository.load_session_data(session.session_id)[1:]
        rr_missing_points, rr_missing_pct = compute_rr_missing_stats(rr_frame)
        hr_missing_points, hr_missing_pct = compute_hr_missing_stats(hr_frame)
        badges = [
            f"Duree {format_duration(session.duree_s)}",
            f"Segments {session.segments_count or 1}",
            format_gap_text(session.gap_durations_s),
            f"FC moy {session.bpm_moyen} bpm",
            f"RR manquants {rr_missing_pct:.1f} %",
            f"FC manquants {hr_missing_pct:.1f} %",
        ]
        note = None
        if session.source_session_ids:
            note = "Sources fusionnees : " + ", ".join(session.source_session_ids)
        render_summary_card(session.annotation or session.session_id, subtitle, badges, note)

        metrics = st.columns(6)
        metrics[0].metric("Date", session.date)
        metrics[1].metric("Debut", start_dt.strftime("%H:%M:%S") if start_dt else "-")
        metrics[2].metric("Fin", end_dt.strftime("%H:%M:%S") if end_dt else "-")
        metrics[3].metric("RR min / max", f"{session.rr_min_ms or '-'} / {session.rr_max_ms or '-'}")
        metrics[4].metric("FC min / max", f"{session.bpm_min} / {session.bpm_max}")
        metrics[5].metric("Mode import", session.import_mode or "-")

    latest_session = sessions[-1]
    if len(sessions) > 1:
        st.info("Plusieurs activites ont ete importees. Les donnees brutes affichees ci-dessous correspondent uniquement a la plus recente.")

    _, rr_frame, hr_frame = repository.load_session_data(latest_session.session_id)
    render_section_label("Donnees brutes importees")
    rr_chart = build_rr_chart_frame(rr_frame) if not rr_frame.empty else pd.DataFrame()
    hr_chart = build_hr_chart_frame(hr_frame) if not hr_frame.empty else pd.DataFrame()

    rr_series = build_chart_series(rr_chart, "rr_interval_ms")
    hr_series = build_chart_series(hr_chart, "bpm")

    if chart_has_values(rr_series, "rr_interval_ms"):
        render_time_series_chart(rr_series, "rr_interval_ms", "RR (ms)", height=240)
    else:
        st.info("Aucune valeur RR affichable sur l'activite la plus recente.")

    if chart_has_values(hr_series, "bpm"):
        render_time_series_chart(hr_series, "bpm", "FC (bpm)", height=240)
    else:
        st.info("Aucune valeur FC affichable sur l'activite la plus recente.")

    with st.expander("Apercu tabulaire de l'activite la plus recente"):
        st.markdown("**RR brut**")
        st.dataframe(rr_frame.head(30), use_container_width=True)
        st.markdown("**FC brute**")
        st.dataframe(hr_frame.head(30), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Import Polar", layout="wide", initial_sidebar_state="expanded")
    inject_styles()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Import multi-segments Polar</div>
            <div class="hero-subtitle">Depose plusieurs fichiers PSL .txt, laisse l'app reconstruire automatiquement une ou plusieurs activites, puis valide l'import.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        render_section_label("Parametres")
        output_dir = st.text_input("Dossier data", value=DEFAULT_OUTPUT_DIR)
        st.caption("Fusion auto < 30 s | confirmation 30-90 s | non-fusion > 90 s")

    render_section_label("Retraitement raw -> processed")
    if st.button("Retraiter les sessions existantes", use_container_width=True):
        importer = PolarImporter("", output_dir)
        reprocess_result = importer.reprocess_saved_sessions()
        st.session_state["reprocess_result"] = reprocess_result

    reprocess_result = st.session_state.get("reprocess_result")
    if reprocess_result:
        if reprocess_result.get("regenerated"):
            st.success("Sessions retrait?es : " + ", ".join(reprocess_result["regenerated"]))
        if reprocess_result.get("skipped"):
            st.info("Sessions non retrait?es : " + ", ".join(reprocess_result["skipped"]))

    uploaded_files = st.file_uploader(
        "Ajoute les fichiers Polar HR et RR (.txt)",
        type=["txt"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        previous_result = st.session_state.get("import_page_result")
        if previous_result:
            render_imported_sessions(output_dir, previous_result["imported_session_ids"])
        st.stop()

    initial_plan = build_plan(uploaded_files, output_dir, None)
    if not initial_plan.activities and not initial_plan.incomplete_segments and not initial_plan.parse_errors:
        st.error("Aucune activite exploitable detectee dans les fichiers fournis.")
        st.stop()

    if initial_plan.ignored_files:
        st.warning("Fichiers ignores : " + ", ".join(initial_plan.ignored_files))
    if initial_plan.incomplete_segments:
        st.warning(
            "Segments incomplets ignores : "
            + " | ".join(
                f"{segment.session_id} (manque {', '.join(segment.missing_types)})"
                for segment in initial_plan.incomplete_segments
            )
        )
    if initial_plan.parse_errors:
        st.error("Segments invalides ignores : " + " | ".join(initial_plan.parse_errors))
    if initial_plan.duplicate_segments:
        st.info(
            "Doublons detectes : "
            + " | ".join(
                f"{duplicate.discarded_session_id} -> {duplicate.kept_session_id} ({duplicate.reason})"
                for duplicate in initial_plan.duplicate_segments
            )
        )

    gap_merge_decisions: dict[str, bool] = {}
    if initial_plan.pending_gap_reviews:
        render_section_label("Confirmations des jonctions 30-90 s")
        for review in initial_plan.pending_gap_reviews:
            radio_key = f"gap_review_{review['decision_key']}"
            decision = st.radio(
                f"Gap {review['gap_s']} s entre {review['prev_session_id']} et {review['next_session_id']}",
                options=["Fusionner ces segments", "Ne pas fusionner"],
                key=radio_key,
            )
            gap_merge_decisions[review["decision_key"]] = decision.startswith("Fusionner")

    plan = build_plan(uploaded_files, output_dir, gap_merge_decisions) if initial_plan.pending_gap_reviews else initial_plan
    if not plan.activities:
        if plan.parse_errors or plan.incomplete_segments:
            st.error("Aucune activite importable: tous les segments fournis sont vides, incomplets ou invalides.")
        else:
            st.error("Aucune activite exploitable apres application des regles de fusion.")
        st.stop()

    render_section_label("Activites detectees")
    selected_activity_ids: list[str] = []
    for activity in plan.activities:
        select_key = f"import_activity_{activity.activity_id}"
        if select_key not in st.session_state:
            st.session_state[select_key] = True

        st.checkbox(
            f"Importer l'activite {activity.activity_index}",
            key=select_key,
            value=st.session_state[select_key],
        )
        if st.session_state.get(select_key, False):
            selected_activity_ids.append(activity.activity_id)

        activity_key = f"activity_name_{activity.activity_id}"
        current_activity_name = st.session_state.get(activity_key, activity.annotation_default)
        render_activity_preview(activity, current_activity_name)
        if activity.gap_reviews:
            gap_frame = pd.DataFrame(activity.gap_reviews)[["prev_session_id", "next_session_id", "gap_s", "status"]]
            st.dataframe(gap_frame, use_container_width=True, hide_index=True)
        st.text_input(
            f"Renommer l'activite {activity.activity_index} avant import",
            value=current_activity_name,
            key=activity_key,
            help="Ce nom sera enregistre comme annotation finale pour l'activite importee.",
        )
        if activity.overlapping_sessions:
            st.checkbox(
                f"Ecraser l'activite deja enregistree qui overlap avec {activity.activity_id}",
                value=False,
                key=f"overwrite_{activity.activity_id}",
            )
        st.divider()

    selected_activities = [activity for activity in plan.activities if activity.activity_id in selected_activity_ids]
    if not selected_activities:
        st.warning("Selectionne au moins une activite a importer.")
    else:
        latest_detected = max(selected_activities, key=lambda activity: activity.t_start)
        render_latest_raw_preview(latest_detected)

    if st.button("Importer les activites detectees", type="primary", use_container_width=True):
        if not selected_activity_ids:
            st.error("Aucune activite selectionnee pour l'import.")
            st.stop()

        annotation_overrides = {
            activity.activity_id: st.session_state.get(f"activity_name_{activity.activity_id}", activity.annotation_default)
            for activity in plan.activities
            if activity.activity_id in selected_activity_ids
        }
        overwrite_activity_ids = {
            activity.activity_id
            for activity in plan.activities
            if activity.activity_id in selected_activity_ids and st.session_state.get(f"overwrite_{activity.activity_id}", False)
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            write_uploaded_files(uploaded_files, temp_dir)
            importer = PolarImporter(temp_dir, output_dir)
            final_plan = importer.plan_import(gap_merge_decisions=gap_merge_decisions)
            final_plan.activities = [activity for activity in final_plan.activities if activity.activity_id in selected_activity_ids]
            result = importer.execute_import_plan(final_plan, annotation_overrides, overwrite_activity_ids)

        st.session_state["import_page_result"] = {
            "imported_session_ids": result.imported_session_ids,
            "overwritten_session_ids": result.overwritten_session_ids,
            "skipped_activity_ids": result.skipped_activity_ids,
        }

        if result.imported_session_ids:
            st.success("Import termine pour : " + ", ".join(result.imported_session_ids))
        if result.overwritten_session_ids:
            st.warning("Activites ecrasees : " + ", ".join(sorted(set(result.overwritten_session_ids))))
        if result.skipped_activity_ids:
            st.info("Activites non importees : " + ", ".join(result.skipped_activity_ids))

    latest_result = st.session_state.get("import_page_result")
    if latest_result:
        render_imported_sessions(output_dir, latest_result["imported_session_ids"])


if __name__ == "__main__":
    main()
