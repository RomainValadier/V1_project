from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from polar_app.repository import ProcessedSessionRepository

try:
    from streamlit_calendar import calendar
except ImportError:
    calendar = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'data'
VISUALS_DIR = PROJECT_ROOT / 'assets' / 'activity_visuals'
SELECTED_EVENT_BACKGROUND = '#f59e0b'
SELECTED_EVENT_BORDER = '#b45309'
SELECTED_EVENT_TEXT = '#1f2937'

TEMPORAL_SEGMENT_COLORS = {
    'echauffement': 'rgba(234, 179, 8, 0.18)',
    'technique': 'rgba(59, 130, 246, 0.18)',
    'randori': 'rgba(220, 38, 38, 0.20)',
    'recuperation': 'rgba(22, 163, 74, 0.18)',
    'retour_calme': 'rgba(139, 92, 246, 0.18)',
    'autre': 'rgba(107, 114, 128, 0.18)',
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(214, 154, 48, 0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(25, 102, 78, 0.10), transparent 28%),
                linear-gradient(180deg, #f7f3e8 0%, #fbfaf6 48%, #edf5f0 100%);
        }
        .block-container {
            max-width: 1500px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            padding: 1.05rem 1.25rem;
            border-radius: 22px;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(242,247,243,0.95));
            border: 1px solid rgba(33, 72, 52, 0.10);
            box-shadow: 0 14px 36px rgba(38, 61, 47, 0.08);
        }
        .hero-title {
            font-size: 1.95rem;
            font-weight: 780;
            color: #173427;
        }
        .hero-subtitle {
            color: #55675c;
            font-size: 1rem;
        }
        .section-chip {
            display: inline-block;
            padding: 0.32rem 0.72rem;
            border-radius: 999px;
            background: rgba(47, 104, 74, 0.10);
            color: #2a5c40;
            font-size: 0.8rem;
            font-weight: 650;
            margin-bottom: 0.55rem;
        }
        .visual-hero {
            display: grid;
            grid-template-columns: minmax(300px, 360px) 1fr;
            gap: 1rem;
            padding: 1rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(239,246,241,0.96));
            border: 1px solid rgba(29, 78, 56, 0.10);
            box-shadow: 0 16px 38px rgba(28, 52, 39, 0.08);
            margin: 0.45rem 0 1rem 0;
        }
        .visual-hero img {
            width: 100%;
            height: 100%;
            min-height: 240px;
            object-fit: cover;
            border-radius: 20px;
            display: block;
        }
        .visual-copy h3 {
            margin: 0;
            font-size: 1.55rem;
            color: #173427;
        }
        .visual-copy p {
            margin: 0.45rem 0 0.8rem 0;
            color: #55675c;
            line-height: 1.5;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 0.9rem;
        }
        .pill {
            padding: 0.34rem 0.72rem;
            border-radius: 999px;
            background: rgba(29, 78, 56, 0.10);
            color: #214834;
            font-size: 0.78rem;
            font-weight: 650;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 0.75rem;
        }
        .summary-card {
            padding: 0.9rem 0.95rem;
            border-radius: 18px;
            background: rgba(250,252,249,0.94);
            border: 1px solid rgba(29, 78, 56, 0.08);
        }
        .summary-label {
            font-size: 0.78rem;
            color: #6b7b72;
            margin-bottom: 0.3rem;
        }
        .summary-value {
            font-size: 1.16rem;
            font-weight: 760;
            color: #173427;
        }
        .soft-note {
            padding: 0.82rem 0.95rem;
            border-radius: 16px;
            background: rgba(240,245,241,0.88);
            border: 1px solid rgba(29, 78, 56, 0.08);
            color: #4d6257;
            font-size: 0.88rem;
            margin-top: 0.9rem;
        }
        @media (max-width: 980px) {
            .visual-hero {
                grid-template-columns: 1fr;
            }
            .summary-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_label(label: str) -> None:
    st.markdown(f'<div class="section-chip">{label}</div>', unsafe_allow_html=True)


def format_duration(duration_seconds: float | int | None) -> str:
    if duration_seconds is None:
        return '-'
    total_seconds = int(round(float(duration_seconds)))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f'{hours} h {minutes:02d} min {seconds:02d} s'
    return f'{minutes} min {seconds:02d} s'


def format_datetime_label(value: datetime | None) -> str:
    return '-' if value is None else value.strftime('%H:%M:%S')

def get_selected_session_id_from_calendar(calendar_state: Any) -> str | None:
    if not isinstance(calendar_state, dict):
        return None
    event_click = calendar_state.get('eventClick')
    if isinstance(event_click, dict):
        event = event_click.get('event') if isinstance(event_click.get('event'), dict) else event_click
        if isinstance(event, dict):
            return (
                event.get('id')
                or ((event.get('extendedProps') or {}).get('session_id') if isinstance(event.get('extendedProps'), dict) else None)
            )
    return None


def highlight_selected_event(events: list[dict[str, Any]], selected_session_id: str | None) -> list[dict[str, Any]]:
    if not selected_session_id:
        return events

    highlighted_events: list[dict[str, Any]] = []
    for event in events:
        event_copy = dict(event)
        event_copy['extendedProps'] = dict(event.get('extendedProps', {}))
        if event_copy.get('id') == selected_session_id:
            event_copy['backgroundColor'] = SELECTED_EVENT_BACKGROUND
            event_copy['borderColor'] = SELECTED_EVENT_BORDER
            event_copy['textColor'] = SELECTED_EVENT_TEXT
            event_copy['extendedProps']['is_selected'] = True
        highlighted_events.append(event_copy)
    return highlighted_events



def svg_data_uri(svg_markup: str) -> str:
    encoded = base64.b64encode(svg_markup.encode('utf-8')).decode('ascii')
    return f'data:image/svg+xml;base64,{encoded}'


def build_training_svg(kind: str, title: str) -> str:
    palette = {
        'randori': ('#ffe7df', '#ef4444', '#991b1b'),
        'technique': ('#e3f0ff', '#3b82f6', '#1d4ed8'),
        'echauffement': ('#fff3d6', '#f59e0b', '#b45309'),
        'recuperation': ('#e5f7ea', '#22c55e', '#15803d'),
        'retour_calme': ('#efe6ff', '#8b5cf6', '#6d28d9'),
        'musculation': ('#ede9fe', '#7c3aed', '#5b21b6'),
        'cardio': ('#ffe6ea', '#f43f5e', '#be123c'),
        'autre': ('#edf2f7', '#64748b', '#334155'),
    }
    background, accent, deep = palette.get(kind, palette['autre'])
    safe_title = html.escape(title)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360"><defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="{background}" /><stop offset="100%" stop-color="#ffffff" /></linearGradient></defs><rect width="640" height="360" rx="28" fill="url(#bg)"/><circle cx="112" cy="90" r="54" fill="{accent}" opacity="0.18"/><circle cx="526" cy="78" r="44" fill="{deep}" opacity="0.14"/><rect x="76" y="228" width="488" height="42" rx="21" fill="{accent}" opacity="0.14"/><path d="M132 212 C180 146, 250 120, 322 144 C380 162, 430 204, 502 186" stroke="{accent}" stroke-width="16" stroke-linecap="round" fill="none" opacity="0.72"/><circle cx="222" cy="155" r="22" fill="{deep}" opacity="0.92"/><rect x="206" y="176" width="34" height="72" rx="17" fill="{deep}" opacity="0.82"/><circle cx="388" cy="170" r="22" fill="{accent}" opacity="0.92"/><rect x="371" y="191" width="34" height="72" rx="17" fill="{accent}" opacity="0.82"/><text x="74" y="74" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#173427">{safe_title}</text><text x="74" y="108" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#52665b">Synthese visuelle de la seance</text></svg>'''
    return svg_data_uri(svg)


def training_visual_key(session) -> str:
    label = str(session.activity_label or '').lower()
    if session.activity_family == 'judo' and session.judo_session_type == 'randoris':
        return 'judo_randoris'
    if session.activity_family == 'judo':
        return 'judo_technique'
    if 'musculation' in label:
        return 'prepa_musculation'
    if 'cardio' in label:
        return 'prepa_cardio'
    return 'autres'


def resolve_visual_source(file_stem: str, fallback_kind: str, title: str) -> str:

    for extension in ('.png', '.jpg', '.jpeg', '.webp'):
        candidate = VISUALS_DIR / f'{file_stem}{extension}'
        if candidate.is_file():
            return str(candidate)
    return build_training_svg(fallback_kind, title)


def randori_feedback_rows(session) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phases_by_index = {
        int(phase.get('phase_index', index)): str(phase.get('phase_label') or phase.get('label') or f'Phase {index + 1}')
        for index, phase in enumerate(session.judo_phases or [])
        if isinstance(phase, dict)
    }
    durations_by_phase_uid: dict[str, list[str]] = {}
    unassigned_durations: list[str] = []
    for segment in session.fc_phase_segments or []:
        if not isinstance(segment, dict):
            continue
        if str(segment.get('label') or '').strip().lower() != 'randori':
            continue
        try:
            duration_label = format_duration(float(segment.get('duration_s', 0.0)))
        except (TypeError, ValueError):
            duration_label = '-'
        phase_uid = segment.get('phase_uid')
        if phase_uid:
            durations_by_phase_uid.setdefault(str(phase_uid), []).append(duration_label)
        else:
            unassigned_durations.append(duration_label)

    for block in session.judo_randori_blocks or []:
        if not isinstance(block, dict):
            continue
        phase_index = int(block.get('phase_index', len(rows)))
        phase_label = phases_by_index.get(phase_index, f'Phase {phase_index + 1}')
        randori_kind = str(block.get('randori_kind') or 'randori')
        phase_uid = block.get('phase_uid')
        phase_durations = list(durations_by_phase_uid.get(str(phase_uid), [])) if phase_uid else []
        entries = [entry for entry in (block.get('randori_entries') or []) if isinstance(entry, dict)]
        entries_by_index = {}
        for entry in entries:
            repetition_index = entry.get('repetition_index')
            if repetition_index is None:
                continue
            try:
                entries_by_index[int(repetition_index)] = entry
            except (TypeError, ValueError):
                continue
        count_value = block.get('randori_count')
        try:
            repetition_count = int(count_value) if count_value not in (None, '', 'NA') else len(entries_by_index)
        except (TypeError, ValueError):
            repetition_count = len(entries_by_index)
        repetition_count = max(repetition_count, len(entries_by_index), len(phase_durations))
        for repetition_index in range(1, repetition_count + 1):
            entry = entries_by_index.get(repetition_index, {})
            if repetition_index - 1 < len(phase_durations):
                randori_duration_label = phase_durations[repetition_index - 1]
            elif unassigned_durations:
                randori_duration_label = unassigned_durations.pop(0)
            else:
                randori_duration_label = '-'
            rows.append(
                {
                    'Phase': phase_label,
                    'Type': randori_kind,
                    'Randori': repetition_index,
                    'Duree': randori_duration_label,
                    'RPE': entry.get('rpe'),
                    'Commentaire': str(entry.get('comment') or '').strip() or '-',
                }
            )
    return rows


def is_session_fully_annotated(session) -> bool:
    if not session.is_activity_annotated:
        return False
    if session.activity_family != 'judo' or session.judo_session_type != 'randoris':
        return True
    if not session.is_temporally_annotated:
        return False
    if getattr(session, 'session_rpe', None) is None:
        return False
    if not session.judo_phases or not session.judo_randori_blocks:
        return False
    for block in session.judo_randori_blocks or []:
        if not isinstance(block, dict):
            return False
        count_value = block.get('randori_count')
        try:
            repetition_count = int(count_value) if count_value not in (None, '', 'NA') else 0
        except (TypeError, ValueError):
            repetition_count = 0
        if repetition_count <= 0:
            return False
        entries = [entry for entry in (block.get('randori_entries') or []) if isinstance(entry, dict)]
        entries_by_index = {}
        for entry in entries:
            repetition_index = entry.get('repetition_index')
            if repetition_index is None:
                continue
            try:
                entries_by_index[int(repetition_index)] = entry
            except (TypeError, ValueError):
                continue
        for repetition_index in range(1, repetition_count + 1):
            entry = entries_by_index.get(repetition_index)
            if not entry or entry.get('rpe') in (None, '', 'NA'):
                return False
    return True


def build_hr_figure(hr_frame: pd.DataFrame, segments: list[dict[str, Any]] | None, total_duration_s: float) -> go.Figure:
    plot_frame = hr_frame.sort_values('t_offset_ms').reset_index(drop=True).copy()
    plot_frame['t_offset_s'] = plot_frame['t_offset_ms'].astype('float64') / 1000.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_frame['t_offset_s'],
            y=plot_frame['bpm'],
            mode='lines',
            line={'color': '#2f6f4e', 'width': 2.6},
            hovertemplate='Temps %{x:.1f}s<br>FC %{y:.0f} bpm<extra></extra>',
            name='FC brute',
        )
    )
    for segment in segments or []:
        color = TEMPORAL_SEGMENT_COLORS.get(str(segment.get('label') or 'autre'), 'rgba(107,114,128,0.16)')
        fig.add_vrect(
            x0=float(segment.get('start_offset_s', 0.0)),
            x1=float(segment.get('end_offset_s', 0.0)),
            fillcolor=color,
            opacity=0.7,
            line_width=1,
            line_color='rgba(30,41,59,0.22)',
            annotation_text=str(segment.get('label') or 'segment'),
            annotation_position='top left',
        )
    fig.update_layout(
        height=410,
        margin={'l': 20, 'r': 20, 't': 24, 'b': 24},
        paper_bgcolor='rgba(255,255,255,0)',
        plot_bgcolor='rgba(255,255,255,0.88)',
        showlegend=False,
        xaxis_title='Temps (s)',
        yaxis_title='FC brute (bpm)',
    )
    fig.add_vline(x=0.0, line_color='rgba(71, 85, 105, 0.35)', line_width=1, line_dash='dash')
    fig.add_vline(x=float(total_duration_s), line_color='rgba(71, 85, 105, 0.35)', line_width=1, line_dash='dash')
    return fig


def render_visual_asset(source: str, caption: str) -> None:
    if source.startswith('data:image/svg+xml;base64,'):
        _, encoded = source.split(',', 1)
        st.image(base64.b64decode(encoded), caption=caption, use_container_width=True)
        return
    st.image(source, caption=caption, use_container_width=True)


def render_visual_overview(session, hr_frame: pd.DataFrame, repository: ProcessedSessionRepository) -> None:
    render_section_label('Synthese finale')
    main_visual_key = training_visual_key(session)
    fallback_kind = 'randori' if session.judo_session_type == 'randoris' else 'technique'
    main_visual = resolve_visual_source(main_visual_key, fallback_kind, session.activity_label or session.annotation or session.session_id)
    start_dt = repository.get_session_start(session)
    end_dt = repository.get_session_end(session)
    hr_valid = hr_frame.loc[hr_frame['bpm'].fillna(0) > 0].copy() if not hr_frame.empty and 'bpm' in hr_frame.columns else pd.DataFrame()
    fc_min = int(hr_valid['bpm'].min()) if not hr_valid.empty else session.bpm_min
    fc_max = int(hr_valid['bpm'].max()) if not hr_valid.empty else session.bpm_max
    feedback_rows = randori_feedback_rows(session)
    feedback_with_rpe = [row for row in feedback_rows if row.get('RPE') is not None]
    mean_randori_rpe = round(sum(int(row['RPE']) for row in feedback_with_rpe) / len(feedback_with_rpe), 1) if feedback_with_rpe else None

    hero_cols = st.columns([1.0, 1.35], gap='large')
    with hero_cols[0]:
        render_visual_asset(main_visual, 'visuel activite')
    with hero_cols[1]:
        st.markdown(
            f'''<div class="visual-copy"><h3>{html.escape(session.annotation or session.session_id)}</h3><p>Lecture finale de la seance, pensee pour comprendre en quelques secondes le contexte de l'entrainement, la dynamique cardiaque et le ressenti associe.</p><div class="pill-row"><span class="pill">{html.escape(session.activity_family or 'famille non renseignee')}</span><span class="pill">{html.escape(session.activity_label or 'activite non renseignee')}</span><span class="pill">RPE seance : {html.escape(str(session.session_rpe) if getattr(session, 'session_rpe', None) is not None else '-')}</span></div><div class="summary-grid"><div class="summary-card"><div class="summary-label">Duree</div><div class="summary-value">{html.escape(format_duration(session.duree_s))}</div></div><div class="summary-card"><div class="summary-label">FC min / max</div><div class="summary-value">{fc_min} / {fc_max}</div></div><div class="summary-card"><div class="summary-label">Debut / fin</div><div class="summary-value">{html.escape(format_datetime_label(start_dt))} - {html.escape(format_datetime_label(end_dt))}</div></div><div class="summary-card"><div class="summary-label">RPE moyen randori</div><div class="summary-value">{mean_randori_rpe if mean_randori_rpe is not None else '-'}</div></div></div><div class="soft-note">{html.escape(session.activity_notes or 'Aucune note generale renseignee pour cette seance.')}</div></div>''',
            unsafe_allow_html=True,
        )

    if session.fc_phase_segments:
        total_duration_s = max(float(session.duree_s or 0.0), float(hr_frame['t_offset_ms'].max()) / 1000.0 if not hr_frame.empty else 0.0)
        fig = build_hr_figure(hr_frame, session.fc_phase_segments, total_duration_s)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    if feedback_rows:
        render_section_label('RPE des randoris')
        st.dataframe(pd.DataFrame(feedback_rows), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title='Visualisation activite', layout='wide', initial_sidebar_state='expanded')
    inject_styles()
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Visualisation activite</div>
            <div class="hero-subtitle">Lecture claire des seances annotees, avec priorite donnee a la comprehension rapide et au ressenti de l'entrainement.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        render_section_label('Parametres')
        output_dir = st.text_input('Dossier data', value=str(DEFAULT_OUTPUT_DIR))
        family_filter = st.selectbox('Famille', options=['toutes', 'judo', 'prepa', 'autres'], index=0)
        annotated_only = st.toggle('Annotees seulement', value=True)

    repository = ProcessedSessionRepository(str(output_dir))
    visible_sessions = repository.list_sessions(include_archived=False)
    if family_filter != 'toutes':
        visible_sessions = [session for session in visible_sessions if session.activity_family == family_filter]
    if annotated_only:
        visible_sessions = [session for session in visible_sessions if session.is_activity_annotated]
    visible_sessions = sorted(visible_sessions, key=lambda session: (session.date, session.heure_debut, session.session_id))

    if not visible_sessions:
        st.info('Aucune seance ne correspond aux filtres actuels.')
        return

    selected_session_id = st.session_state.get('selected_visual_activity_id')
    if selected_session_id not in {session.session_id for session in visible_sessions}:
        selected_session_id = visible_sessions[-1].session_id
        st.session_state['selected_visual_activity_id'] = selected_session_id

    events = highlight_selected_event(
        repository.build_calendar_events(
            include_archived=False,
            family_filter=family_filter,
            annotated_only=annotated_only,
        ),
        selected_session_id,
    )

    render_section_label('Calendrier')
    if calendar is None:
        st.warning("Le module `streamlit-calendar` n'est pas installe localement. La page utilise temporairement une selection simple.")
        fallback_options = {
            f"{session.annotation or session.session_id} | {session.date} {session.heure_debut}": session.session_id
            for session in visible_sessions
        }
        option_labels = list(fallback_options.keys())
        option_values = list(fallback_options.values())
        default_index = option_values.index(selected_session_id) if selected_session_id in option_values else len(option_labels) - 1
        selected_label = st.selectbox('Activite', options=option_labels, index=default_index)
        selected_session_id = fallback_options[selected_label]
        st.session_state['selected_visual_activity_id'] = selected_session_id
    else:
        calendar_options = {
            'initialView': 'dayGridMonth',
            'headerToolbar': {
                'left': 'today prev,next',
                'center': 'title',
                'right': 'dayGridMonth,timeGridWeek,timeGridDay',
            },
            'height': 680,
            'locale': 'fr',
            'editable': False,
            'selectable': False,
            'eventDisplay': 'block',
        }
        calendar_state = calendar(
            events=events,
            options=calendar_options,
            custom_css="""
                .fc-event {
                    border-radius: 10px;
                    padding: 2px 4px;
                    font-size: 0.78rem;
                }
            """,
            key='visual_activity_calendar',
        )
        clicked_session_id = get_selected_session_id_from_calendar(calendar_state)
        if clicked_session_id and clicked_session_id != st.session_state.get('selected_visual_activity_id'):
            st.session_state['selected_visual_activity_id'] = clicked_session_id
            st.rerun()

    selected_session_id = st.session_state.get('selected_visual_activity_id', selected_session_id)
    session = repository.get_sessions([selected_session_id], include_archived=True)[0]
    _, _, hr_frame = repository.load_session_data(selected_session_id)

    if not is_session_fully_annotated(session):
        st.info("Cette seance n'est pas completement renseignee, mais elle reste visible ici des lors que l'activite a ete annotee.")

    render_visual_overview(session, hr_frame, repository)


if __name__ == '__main__':
    main()


