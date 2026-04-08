from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime
from typing import Any

import altair as alt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from polar_app.analytics import build_hr_chart_frame
from polar_app.fc_segment_component import fc_segment_editor
from polar_app.repository import ProcessedSessionRepository

try:
    from streamlit_calendar import calendar
except ImportError:
    calendar = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")
ACTIVITY_REFERENCE_PATH = os.path.join(PROJECT_ROOT, "liste_activite.txt")
JUDO_PHASE_REFERENCE_PATH = os.path.join(PROJECT_ROOT, "seance_judo_phase.txt")
RANDORI_KIND_OPTIONS = ["randori_tw", "randori_nw"]
RANDORI_DURATION_OPTIONS = [f"{value:.1f}" for value in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]]
RANDORI_COUNT_OPTIONS = ["NA"] + [str(value) for value in range(1, 31)]
RANDORI_REST_OPTIONS = ["NA"] + [f"{value:.1f}" for value in [index / 2 for index in range(1, 21)]]
RPE_OPTIONS = ["NA"] + [str(value) for value in range(1, 11)]
RANDORI_PHASE_LABELS = {"randori_tw", "randori_nw"}
TEMPORAL_SEGMENT_LABELS = ["echauffement", "technique", "randori_tw", "randori_nw", "recuperation", "retour_calme"]
TEMPORAL_SEGMENT_COLORS = {
    "echauffement": "rgba(93,202,165,0.3)",
    "technique": "rgba(206,203,246,0.3)",
    "randori_tw": "rgba(240,153,123,0.3)",
    "randori_nw": "rgba(237,147,177,0.3)",
    "recuperation": "rgba(133,183,235,0.3)",
    "retour_calme": "rgba(211,209,199,0.3)",
}
TEMPORAL_SEGMENT_HEX_COLORS = {
    "echauffement": "#1D9E75",
    "technique": "#534AB7",
    "randori_tw": "#D85A30",
    "randori_nw": "#D4537E",
    "recuperation": "#378ADD",
    "retour_calme": "#888780",
}
SELECTED_EVENT_BACKGROUND = "#f59e0b"
SELECTED_EVENT_BORDER = "#b45309"
SELECTED_EVENT_TEXT = "#1f2937"
SEGMENT_UI_PALETTE = {
    "echauffement": {"bg": "rgba(93,202,165,0.3)", "border": "#1D9E75", "text": "#085041"},
    "randori_tw": {"bg": "rgba(240,153,123,0.3)", "border": "#D85A30", "text": "#712B13"},
    "randori_nw": {"bg": "rgba(237,147,177,0.3)", "border": "#D4537E", "text": "#72243E"},
    "recuperation": {"bg": "rgba(133,183,235,0.3)", "border": "#378ADD", "text": "#0C447C"},
    "technique": {"bg": "rgba(206,203,246,0.3)", "border": "#534AB7", "text": "#3C3489"},
    "retour_calme": {"bg": "rgba(211,209,199,0.3)", "border": "#888780", "text": "#444441"},
}
FC_SEGMENT_EDITOR = fc_segment_editor


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
            max-width: 1550px;
            padding-top: 1.3rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            padding: 1.05rem 1.25rem;
            border-radius: 20px;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, rgba(255,255,255,0.97), rgba(242,247,243,0.95));
            border: 1px solid rgba(33, 72, 52, 0.10);
            box-shadow: 0 14px 36px rgba(38, 61, 47, 0.08);
        }
        .hero-title {
            font-size: 1.9rem;
            font-weight: 760;
            color: #173427;
        }
        .hero-subtitle {
            color: #55675c;
            font-size: 1rem;
        }
        .section-chip {
            display: inline-block;
            padding: 0.3rem 0.72rem;
            border-radius: 999px;
            background: rgba(47, 104, 74, 0.10);
            color: #2a5c40;
            font-size: 0.8rem;
            font-weight: 650;
            margin-bottom: 0.5rem;
        }
        .status-row {
            display: flex;
            gap: 0.5rem;
            margin: 0.35rem 0 0.85rem 0;
            flex-wrap: wrap;
        }
        .status-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.28rem 0.7rem;
            font-size: 0.78rem;
            font-weight: 650;
            background: rgba(33, 72, 52, 0.10);
            color: #214834;
        }
        .status-pill.warn {
            background: rgba(185, 28, 28, 0.10);
            color: #991b1b;
        }
        .status-pill.muted {
            background: rgba(100, 116, 139, 0.12);
            color: #475569;
        }
        .detail-card {
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(33, 56, 42, 0.08);
            border-radius: 18px;
            padding: 1rem 1rem;
            box-shadow: 0 10px 26px rgba(41, 60, 47, 0.05);
            margin-bottom: 1rem;
        }
        div.stButton > button[kind=\"primary\"] {
            background-color: #1D9E75;
            color: white;
            font-weight: 500;
            padding: 10px 20px;
            font-size: 14px;
            border: none;
        }
        div.stButton > button[kind=\"secondary\"] {
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_label(label: str) -> None:
    st.markdown(f'<div class="section-chip">{label}</div>', unsafe_allow_html=True)


def terminal_category_label(raw_label: str | None) -> str:
    if not raw_label:
        return "-"
    parts = [part.strip() for part in str(raw_label).split(">") if part.strip()]
    return parts[-1] if parts else str(raw_label)


def segment_palette(label: str | None) -> dict[str, str]:
    normalized = str(label or "technique").strip().lower()
    return SEGMENT_UI_PALETTE.get(normalized, SEGMENT_UI_PALETTE["technique"])


def build_badge_html(label: str, *, tone: str = "default") -> str:
    tone_map = {
        "default": ("rgba(33,72,52,0.10)", "#214834"),
        "warn": ("rgba(185,28,28,0.10)", "#991b1b"),
        "muted": ("rgba(100,116,139,0.12)", "#475569"),
        "success": ("rgba(29,158,117,0.14)", "#085041"),
    }
    background, color = tone_map.get(tone, tone_map["default"])
    return f'<span style="display:inline-block;border-radius:999px;padding:0.32rem 0.78rem;font-size:0.78rem;font-weight:650;background:{background};color:{color};">{label}</span>'


def build_segment_chip_html(label: str, suffix: str = "") -> str:
    palette = segment_palette(label)
    safe_label = terminal_category_label(label).replace('_', ' ')
    suffix_markup = f" <span style='opacity:0.7'>{suffix}</span>" if suffix else ""
    return (
        f'<div style="display:inline-flex;align-items:center;gap:0.35rem;padding:0.38rem 0.72rem;border-radius:999px;'
        f'background:{palette["bg"]};border:1px solid {palette["border"]};color:{palette["text"]};font-size:0.8rem;font-weight:600;">'
        f'{safe_label}{suffix_markup}</div>'
    )


def build_metric_card_html(label: str, value: str, unit: str = "") -> str:
    unit_markup = f" <span style='font-size:0.74rem;color:#667066;font-weight:500'>{unit}</span>" if unit else ""
    return (
        "<div style='background:#f5f5f3;border-radius:8px;padding:12px;min-height:78px;'>"
        f"<div style='font-size:11px;color:#6b7280;margin-bottom:6px;'>{label}</div>"
        f"<div style='font-size:18px;font-weight:500;color:#163427;'>{value}{unit_markup}</div>"
        "</div>"
    )


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def temporal_history_key(session_id: str) -> str:
    return f"fc_segments_history_{session_id}"


def temporal_history_index_key(session_id: str) -> str:
    return f"fc_segments_history_idx_{session_id}"


def reset_temporal_history(session_id: str, segments: list[dict[str, Any]]) -> None:
    st.session_state[temporal_history_key(session_id)] = [deepcopy(segments)]
    st.session_state[temporal_history_index_key(session_id)] = 0


def push_temporal_history(session_id: str, segments: list[dict[str, Any]]) -> None:
    history_key = temporal_history_key(session_id)
    history_index_key = temporal_history_index_key(session_id)
    history = list(st.session_state.get(history_key, []))
    current_index = int(st.session_state.get(history_index_key, -1))
    snapshot = deepcopy(segments)
    if history and current_index >= 0 and history[current_index] == snapshot:
        return
    history = history[: current_index + 1]
    history.append(snapshot)
    st.session_state[history_key] = history
    st.session_state[history_index_key] = len(history) - 1


def restore_temporal_history(session_id: str, repository: ProcessedSessionRepository, total_duration_s: float, direction: int) -> bool:
    history = list(st.session_state.get(temporal_history_key(session_id), []))
    if not history:
        return False
    current_index = int(st.session_state.get(temporal_history_index_key(session_id), 0))
    target_index = current_index + direction
    if target_index < 0 or target_index >= len(history):
        return False
    restored_segments = repository.normalize_fc_phase_segments(deepcopy(history[target_index]), total_duration_s)
    st.session_state[temporal_history_index_key(session_id)] = target_index
    st.session_state[temporal_segments_key(session_id)] = restored_segments
    st.session_state[temporal_selected_segment_key(session_id)] = 0 if restored_segments else None
    repository.save_fc_phase_segments(session_id, restored_segments)
    return True


def compute_segment_fc_mean(hr_frame: pd.DataFrame, segment: dict[str, Any] | None) -> str:
    if segment is None or hr_frame.empty or 'bpm' not in hr_frame.columns:
        return '-'
    start_s = float(segment.get('start_offset_s', 0.0))
    end_s = float(segment.get('end_offset_s', start_s))
    frame = hr_frame.copy()
    frame['t_offset_s'] = frame['t_offset_ms'].astype('float64') / 1000.0
    window = frame.loc[(frame['t_offset_s'] >= start_s) & (frame['t_offset_s'] <= end_s) & frame['bpm'].notna()]
    if window.empty:
        return '-'
    return f"{int(round(float(window['bpm'].mean())))} bpm"


def split_selected_temporal_segment(session_id: str, new_label: str, total_duration_s: float, repository: ProcessedSessionRepository, side: str) -> bool:
    selected_index = st.session_state.get(temporal_selected_segment_key(session_id))
    segments = [dict(segment) for segment in st.session_state.get(temporal_segments_key(session_id), [])]
    if selected_index is None or not (0 <= int(selected_index) < len(segments)):
        return False
    selected_index = int(selected_index)
    segment = dict(segments[selected_index])
    start_s = float(segment.get('start_offset_s', 0.0))
    end_s = float(segment.get('end_offset_s', start_s))
    duration_s = end_s - start_s
    min_duration_s = float(repository.MIN_SEGMENT_DURATION_S)
    if duration_s < min_duration_s * 2:
        st.session_state['activity_management_notice'] = {'level': 'warning', 'message': 'Le segment est trop court pour etre decoupe.'}
        return False
    split_ratio = 0.3 if side == 'left' else 0.7
    split_point = start_s + duration_s * split_ratio
    split_point = max(start_s + min_duration_s, min(end_s - min_duration_s, split_point))
    if side == 'left':
        new_segment = {'label': new_label, 'start_offset_s': start_s, 'end_offset_s': split_point, 'source': 'manual', 'phase_uid': None, 'locked': False}
        segment['start_offset_s'] = split_point
        segments[selected_index] = segment
        segments.insert(selected_index, new_segment)
        new_index = selected_index
    else:
        new_segment = {'label': new_label, 'start_offset_s': split_point, 'end_offset_s': end_s, 'source': 'manual', 'phase_uid': None, 'locked': False}
        segment['end_offset_s'] = split_point
        segments[selected_index] = segment
        segments.insert(selected_index + 1, new_segment)
        new_index = selected_index + 1
    if not persist_temporal_segments(session_id, segments, total_duration_s, repository):
        return False
    st.session_state[temporal_selected_segment_key(session_id)] = new_index
    return True


def render_notice() -> None:
    notice = st.session_state.pop("activity_management_notice", None)
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


def load_reference_lines(path: str) -> list[str]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8-sig") as file_obj:
        return [line.strip() for line in file_obj.readlines() if line.strip()]


def format_duration(duration_seconds: float | int | None) -> str:
    if duration_seconds is None:
        return "-"
    total_seconds = int(round(float(duration_seconds)))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min {seconds:02d} s"
    return f"{minutes} min {seconds:02d} s"


def format_datetime_label(value: datetime | None) -> str:
    return "-" if value is None else value.strftime("%H:%M:%S")


def activity_label_options(activity_family: str | None, activity_options: list[str], judo_session_type: str | None) -> list[str]:
    if not activity_family:
        return activity_options
    filtered = [item for item in activity_options if item == activity_family or item.startswith(f"{activity_family} >")]
    if activity_family != "judo":
        return filtered
    if judo_session_type == "technique":
        return [item for item in filtered if item == "judo > technique"]
    if judo_session_type == "randoris":
        return [item for item in filtered if item.startswith("judo > randoris >")]
    return filtered


def phase_is_randori(phase_label: str) -> bool:
    return str(phase_label).strip().lower() in RANDORI_PHASE_LABELS


def phase_category(phase_label: str) -> str:
    return str(phase_label).strip().lower()


def saved_phase_labels(session) -> list[str]:
    labels: list[str] = []
    for phase in session.judo_phases or []:
        if not isinstance(phase, dict):
            continue
        label = str(phase.get("phase_label") or phase.get("label") or "").strip()
        if label:
            labels.append(label)
    return labels


def current_phase_labels(session_id: str) -> list[str]:
    labels: list[str] = []
    for item in st.session_state.get(f"phase_items_{session_id}", []):
        phase_uid = item.get("phase_uid")
        fallback_label = item.get("phase_label") or "echauffement"
        labels.append(str(st.session_state.get(f"phase_label_{session_id}_{phase_uid}", fallback_label)))
    return labels


def default_randori_kind(phase_label: str) -> str:
    if phase_label == "randori_nw":
        return "randori_nw"
    return "randori_tw"


def to_optional_number(raw_value: str | None, integer: bool = False) -> str | int | float:
    if raw_value in (None, "", "NA"):
        return "NA"
    return int(raw_value) if integer else float(raw_value)


def to_optional_int(raw_value: str | None) -> int | None:
    if raw_value in (None, "", "NA"):
        return None
    return int(raw_value)


def to_optional_text(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    return text or None


def highlight_selected_event(events: list[dict[str, Any]], selected_session_id: str | None) -> list[dict[str, Any]]:
    if not selected_session_id:
        return events

    highlighted_events: list[dict[str, Any]] = []
    for event in events:
        event_copy = dict(event)
        event_copy["extendedProps"] = dict(event.get("extendedProps", {}))
        if event_copy.get("id") == selected_session_id:
            event_copy["backgroundColor"] = SELECTED_EVENT_BACKGROUND
            event_copy["borderColor"] = SELECTED_EVENT_BORDER
            event_copy["textColor"] = SELECTED_EVENT_TEXT
            event_copy["extendedProps"]["is_selected"] = True
        highlighted_events.append(event_copy)
    return highlighted_events


def temporal_segments_key(session_id: str) -> str:
    return f"fc_segments_state_{session_id}"


def temporal_selected_segment_key(session_id: str) -> str:
    return f"fc_selected_segment_{session_id}"


def temporal_selected_point_key(session_id: str) -> str:
    return f"fc_graph_point_{session_id}"


def temporal_edit_mode_key(session_id: str) -> str:
    return f"fc_graph_edit_mode_{session_id}"


def temporal_move_anchor_key(session_id: str) -> str:
    return f"fc_graph_move_anchor_{session_id}"


def temporal_last_graph_action_key(session_id: str) -> str:
    return f"fc_graph_last_action_{session_id}"


def temporal_component_event_key(session_id: str) -> str:
    return f"fc_component_event_{session_id}"


def format_offset_label(offset_s: float | int | None) -> str:
    if offset_s is None:
        return "-"
    total_seconds = int(round(float(offset_s)))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def get_selected_graph_point(plotly_state: Any) -> dict[str, float] | None:
    if not isinstance(plotly_state, dict):
        return None
    selection = plotly_state.get("selection") if isinstance(plotly_state.get("selection"), dict) else plotly_state
    if not isinstance(selection, dict):
        return None
    points = selection.get("points")
    if not points:
        return None
    point = points[0]
    x_value = point.get("x")
    y_value = point.get("y")
    if x_value is None:
        return None
    return {"offset_s": float(x_value), "bpm": float(y_value) if y_value is not None else None}


def temporal_status_label(segments: list[dict[str, Any]]) -> str:
    if not segments:
        return "non annotee"
    if all(str(segment.get("source", "manual")) == "auto" for segment in segments):
        return "auto-generee non ajustee"
    return "annotee temporellement"


def serialize_temporal_segments_for_component(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        label = str(segment.get("label") or "technique")
        serialized.append(
            {
                "id": int(segment.get("segment_index", index)),
                "type": label,
                "t_debut_s": round(float(segment.get("start_offset_s", 0.0)), 3),
                "t_fin_s": round(float(segment.get("end_offset_s", 0.0)), 3),
                "color": TEMPORAL_SEGMENT_HEX_COLORS.get(label, TEMPORAL_SEGMENT_HEX_COLORS["technique"]),
            }
        )
    return serialized


def deserialize_temporal_segments_from_component(raw_segments: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    deserialized: list[dict[str, Any]] = []
    for index, segment in enumerate(raw_segments or []):
        label = str(segment.get("type") or segment.get("label") or "technique")
        deserialized.append(
            {
                "segment_index": int(segment.get("id", index)),
                "label": label,
                "start_offset_s": float(segment.get("t_debut_s", 0.0)),
                "end_offset_s": float(segment.get("t_fin_s", 0.0)),
                "duration_s": max(float(segment.get("t_fin_s", 0.0)) - float(segment.get("t_debut_s", 0.0)), 0.0),
                "source": "manual",
                "phase_uid": None,
                "locked": False,
            }
        )
    return deserialized


def build_hr_points_for_component(hr_frame: pd.DataFrame) -> list[dict[str, float]]:
    if hr_frame.empty:
        return []
    plot_frame = hr_frame.sort_values("t_offset_ms").reset_index(drop=True).copy()
    plot_frame["t_offset_s"] = plot_frame["t_offset_ms"].astype("float64") / 1000.0
    return [
        {"t_offset_s": float(row.t_offset_s), "bpm": float(row.bpm)}
        for row in plot_frame[["t_offset_s", "bpm"]].itertuples(index=False)
        if pd.notna(row.bpm)
    ]


def build_plotly_hr_figure(
    hr_frame: pd.DataFrame,
    segments: list[dict[str, Any]],
    selected_segment_index: int | None,
    total_duration_s: float,
) -> go.Figure:
    plot_frame = hr_frame.sort_values("t_offset_ms").reset_index(drop=True).copy()
    plot_frame["t_offset_s"] = plot_frame["t_offset_ms"].astype("float64") / 1000.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_frame["t_offset_s"],
            y=plot_frame["bpm"],
            mode="lines+markers",
            line={"color": "#2f6f4e", "width": 2.5},
            marker={"size": 4.5, "color": "#204f39"},
            name="FC brute",
            hovertemplate="Temps %{x:.1f}s<br>FC %{y:.0f} bpm<extra></extra>",
        )
    )
    fig.add_vline(x=0.0, line_color="rgba(71, 85, 105, 0.45)", line_width=1, line_dash="dash")
    fig.add_vline(x=float(total_duration_s), line_color="rgba(71, 85, 105, 0.45)", line_width=1, line_dash="dash")
    for segment in segments:
        is_selected = selected_segment_index is not None and int(segment.get("segment_index", -1)) == int(selected_segment_index)
        palette = segment_palette(str(segment.get("label") or "technique"))
        label = str(segment.get("label", "segment")).replace("_", " ")
        fig.add_vrect(
            x0=float(segment.get("start_offset_s", 0.0)),
            x1=float(segment.get("end_offset_s", 0.0)),
            fillcolor=palette["bg"],
            opacity=0.45 if is_selected else 0.25,
            layer="below",
            line_width=2 if is_selected else 1,
            line_color=palette["border"] if is_selected else "rgba(30,41,59,0.24)",
            annotation_text=f"{label}{' *' if is_selected else ''}".strip(),
            annotation_position="top left",
        )
    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=460,
        showlegend=False,
        dragmode="select",
        xaxis_title="Temps (s)",
        yaxis_title="FC brute (bpm)",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.85)",
    )
    return fig


def set_temporal_segments_state(session_id: str, segments: list[dict[str, Any]], total_duration_s: float, repository: ProcessedSessionRepository) -> None:
    normalized = repository.normalize_fc_phase_segments(segments, total_duration_s)
    st.session_state[temporal_segments_key(session_id)] = normalized
    if not normalized:
        st.session_state[temporal_selected_segment_key(session_id)] = None
    else:
        current_index = st.session_state.get(temporal_selected_segment_key(session_id))
        max_index = len(normalized) - 1
        st.session_state[temporal_selected_segment_key(session_id)] = 0 if current_index is None else min(int(current_index), max_index)


def persist_temporal_segments(
    session_id: str,
    segments: list[dict[str, Any]],
    total_duration_s: float,
    repository: ProcessedSessionRepository,
    *,
    anchor_index: int | None = None,
    push_history_snapshot: bool = True,
) -> bool:
    try:
        if anchor_index is None:
            saved_segments = repository.save_fc_phase_segments(session_id, segments)
        else:
            saved_segments = repository.save_fc_phase_segments_for_edit(
                session_id,
                segments,
                anchor_index=anchor_index,
            )
    except Exception as exc:
        st.session_state["activity_management_notice"] = {"level": "error", "message": str(exc)}
        return False
    set_temporal_segments_state(session_id, saved_segments, total_duration_s, repository)
    if push_history_snapshot:
        push_temporal_history(session_id, saved_segments)
    return True


def add_temporal_segment(
    session_id: str,
    label: str,
    total_duration_s: float,
    repository: ProcessedSessionRepository,
    *,
    side: str = "right",
) -> bool:
    segments = [dict(segment) for segment in st.session_state.get(temporal_segments_key(session_id), [])]
    selected_index = st.session_state.get(temporal_selected_segment_key(session_id))
    min_duration_s = float(repository.MIN_SEGMENT_DURATION_S)
    default_duration_s = max(60.0, min_duration_s)

    if selected_index is None or not (0 <= int(selected_index) < len(segments)):
        st.session_state["activity_management_notice"] = {
            "level": "warning",
            "message": "Selectionne d'abord un segment avant d'en ajouter un nouveau.",
        }
        return False

    selected_index = int(selected_index)
    selected_segment = dict(segments[selected_index])
    selected_start_s = float(selected_segment.get("start_offset_s", 0.0))
    selected_end_s = float(selected_segment.get("end_offset_s", 0.0))
    selected_duration_s = selected_end_s - selected_start_s

    if side == "left":
        insert_at = selected_index
        if selected_index > 0:
            previous_segment = dict(segments[selected_index - 1])
            previous_start_s = float(previous_segment.get("start_offset_s", 0.0))
            previous_end_s = float(previous_segment.get("end_offset_s", selected_start_s))
            previous_duration_s = previous_end_s - previous_start_s
            available_s = previous_duration_s - min_duration_s
            if available_s < min_duration_s:
                st.session_state["activity_management_notice"] = {
                    "level": "warning",
                    "message": "Le segment precedent est trop court pour inserer un nouveau segment a gauche.",
                }
                return False
            new_duration_s = min(default_duration_s, available_s)
            new_segment = {
                "label": label,
                "start_offset_s": selected_start_s - new_duration_s,
                "end_offset_s": selected_start_s,
                "source": "manual",
                "phase_uid": None,
                "locked": False,
            }
            previous_segment["end_offset_s"] = selected_start_s - new_duration_s
            previous_segment["source"] = "manual"
            segments[selected_index - 1] = previous_segment
            segments.insert(insert_at, new_segment)
        else:
            available_s = selected_duration_s - min_duration_s
            if available_s < min_duration_s:
                st.session_state["activity_management_notice"] = {
                    "level": "warning",
                    "message": "Le segment actif est trop court pour inserer un nouveau segment a gauche.",
                }
                return False
            new_duration_s = min(default_duration_s, available_s)
            new_segment = {
                "label": label,
                "start_offset_s": selected_start_s,
                "end_offset_s": selected_start_s + new_duration_s,
                "source": "manual",
                "phase_uid": None,
                "locked": False,
            }
            selected_segment["start_offset_s"] = selected_start_s + new_duration_s
            selected_segment["source"] = "manual"
            segments[selected_index] = selected_segment
            segments.insert(insert_at, new_segment)
    else:
        insert_at = selected_index + 1
        if insert_at < len(segments):
            next_segment = dict(segments[insert_at])
            next_start_s = float(next_segment.get("start_offset_s", selected_end_s))
            next_end_s = float(next_segment.get("end_offset_s", next_start_s))
            next_duration_s = next_end_s - next_start_s
            available_s = next_duration_s - min_duration_s
            if available_s < min_duration_s:
                st.session_state["activity_management_notice"] = {
                    "level": "warning",
                    "message": "Le segment suivant est trop court pour inserer un nouveau segment a droite.",
                }
                return False
            new_duration_s = min(default_duration_s, available_s)
            new_segment = {
                "label": label,
                "start_offset_s": selected_end_s,
                "end_offset_s": selected_end_s + new_duration_s,
                "source": "manual",
                "phase_uid": None,
                "locked": False,
            }
            next_segment["start_offset_s"] = selected_end_s + new_duration_s
            next_segment["source"] = "manual"
            segments[insert_at] = next_segment
            segments.insert(insert_at, new_segment)
        else:
            available_s = selected_duration_s - min_duration_s
            if available_s < min_duration_s:
                st.session_state["activity_management_notice"] = {
                    "level": "warning",
                    "message": "Le segment actif est trop court pour creer un nouveau segment en fin de seance.",
                }
                return False
            new_duration_s = min(default_duration_s, available_s)
            selected_segment["end_offset_s"] = selected_end_s - new_duration_s
            selected_segment["source"] = "manual"
            new_segment = {
                "label": label,
                "start_offset_s": selected_end_s - new_duration_s,
                "end_offset_s": selected_end_s,
                "source": "manual",
                "phase_uid": None,
                "locked": False,
            }
            segments[selected_index] = selected_segment
            segments.append(new_segment)

    if not persist_temporal_segments(session_id, segments, total_duration_s, repository):
        return False
    st.session_state[temporal_selected_segment_key(session_id)] = insert_at
    side_label = "gauche" if side == "left" else "droite"
    st.session_state["activity_management_notice"] = {
        "level": "success",
        "message": f"Nouveau segment ajoute a {side_label} du segment selectionne.",
    }
    return True


def delete_temporal_segment(session_id: str, total_duration_s: float, repository: ProcessedSessionRepository) -> bool:
    selected_index = st.session_state.get(temporal_selected_segment_key(session_id))
    if selected_index is None:
        return False
    segments = [dict(segment) for segment in st.session_state.get(temporal_segments_key(session_id), [])]
    if not (0 <= int(selected_index) < len(segments)):
        return False

    selected_index = int(selected_index)
    selected_segment = dict(segments[selected_index])
    selected_start_s = float(selected_segment.get("start_offset_s", 0.0))
    selected_end_s = float(selected_segment.get("end_offset_s", 0.0))

    if len(segments) == 1:
        if not persist_temporal_segments(session_id, [], total_duration_s, repository):
            return False
        st.session_state[temporal_selected_segment_key(session_id)] = None
        st.session_state["activity_management_notice"] = {
            "level": "success",
            "message": "Le dernier segment a ete supprime.",
        }
        return True

    if selected_index > 0:
        previous_segment = dict(segments[selected_index - 1])
        previous_segment["end_offset_s"] = selected_end_s
        previous_segment["source"] = "manual"
        segments[selected_index - 1] = previous_segment
        segments.pop(selected_index)
        next_selected_index = selected_index - 1
    else:
        next_segment = dict(segments[1])
        next_segment["start_offset_s"] = selected_start_s
        next_segment["source"] = "manual"
        segments[1] = next_segment
        segments.pop(0)
        next_selected_index = 0

    if not persist_temporal_segments(session_id, segments, total_duration_s, repository):
        return False
    st.session_state[temporal_selected_segment_key(session_id)] = min(next_selected_index, len(st.session_state.get(temporal_segments_key(session_id), [])) - 1)
    st.session_state["activity_management_notice"] = {
        "level": "success",
        "message": "Le segment selectionne a ete supprime et la timeline a ete recalee.",
    }
    return True


def update_selected_temporal_segment(
    session_id: str,
    total_duration_s: float,
    repository: ProcessedSessionRepository,
    *,
    label: str | None = None,
    start_offset_s: float | None = None,
    end_offset_s: float | None = None,
    move_to_offset_s: float | None = None,
    move_anchor: str = "start",
    delta_s: float | None = None,
) -> bool:
    selected_index = st.session_state.get(temporal_selected_segment_key(session_id))
    if selected_index is None:
        return False
    segments = list(st.session_state.get(temporal_segments_key(session_id), []))
    if not (0 <= int(selected_index) < len(segments)):
        return False
    target_index = int(selected_index)
    updated_segment = dict(segments[target_index])
    current_start = float(updated_segment.get("start_offset_s", 0.0))
    current_end = float(updated_segment.get("end_offset_s", current_start + 1.0))
    duration_s = max(float(updated_segment.get("duration_s", current_end - current_start)), 1.0)

    if label is not None:
        updated_segment["label"] = label
    if delta_s is not None:
        current_start += float(delta_s)
        current_end += float(delta_s)
    if move_to_offset_s is not None:
        target_offset_s = float(move_to_offset_s)
        if move_anchor == "centre":
            current_start = target_offset_s - duration_s / 2.0
        else:
            current_start = target_offset_s
        current_end = current_start + duration_s
    if start_offset_s is not None:
        current_start = float(start_offset_s)
    if end_offset_s is not None:
        current_end = float(end_offset_s)

    updated_segment["start_offset_s"] = current_start
    updated_segment["end_offset_s"] = current_end
    updated_segment["source"] = "manual"
    segments[target_index] = updated_segment
    if not persist_temporal_segments(session_id, segments, total_duration_s, repository, anchor_index=target_index):
        return False
    st.session_state[temporal_selected_segment_key(session_id)] = min(target_index, len(st.session_state.get(temporal_segments_key(session_id), [])) - 1)
    return True


def auto_generate_temporal_segments(session_id: str, repository: ProcessedSessionRepository, total_duration_s: float) -> bool:
    try:
        generated_segments = repository.generate_default_fc_phase_segments(session_id)
        saved_segments = repository.save_fc_phase_segments(session_id, generated_segments)
    except Exception as exc:
        st.session_state['activity_management_notice'] = {
            'level': 'warning',
            'message': f'Auto-segmentation impossible : {exc}',
        }
        return False

    set_temporal_segments_state(session_id, saved_segments, total_duration_s, repository)
    reset_temporal_history(session_id, saved_segments)
    st.session_state[temporal_selected_segment_key(session_id)] = 0 if saved_segments else None
    st.session_state['activity_management_notice'] = {
        'level': 'success',
        'message': 'Auto-segmentation generee. Les segments sont prets a etre ajustes sur le graphe.',
    }
    return True


def render_temporal_annotation_module(session, hr_frame: pd.DataFrame, repository: ProcessedSessionRepository) -> None:
    session_id = session.session_id
    total_duration_s = max(float(session.duree_s or 0.0), float(hr_frame['t_offset_ms'].max()) / 1000.0 if not hr_frame.empty else 0.0)
    segments = st.session_state.get(temporal_segments_key(session_id), [])
    selected_segment_index = st.session_state.get(temporal_selected_segment_key(session_id))
    active_segment = None
    if segments and selected_segment_index is not None and 0 <= int(selected_segment_index) < len(segments):
        active_segment = segments[int(selected_segment_index)]

    history = st.session_state.get(temporal_history_key(session_id), [])
    history_index = int(st.session_state.get(temporal_history_index_key(session_id), len(history) - 1 if history else 0))
    can_undo = bool(history) and history_index > 0
    can_redo = bool(history) and history_index < len(history) - 1

    render_section_label('Segmentation temporelle')
    top_bar = st.columns([2.8, 1.2])
    with top_bar[0]:
        if active_segment is None:
            st.markdown(build_badge_html('Aucun segment selectionne', tone='muted'), unsafe_allow_html=True)
        else:
            label = str(active_segment.get('label') or 'technique')
            summary = f"{format_offset_label(active_segment.get('start_offset_s'))} -> {format_offset_label(active_segment.get('end_offset_s'))} ({format_duration(active_segment.get('duration_s'))})"
            st.markdown(build_segment_chip_html(label, summary), unsafe_allow_html=True)
    with top_bar[1]:
        action_cols = st.columns(4)
        if action_cols[0].button('Annuler', disabled=not can_undo, use_container_width=True, key=f'undo_seg_{session_id}'):
            if restore_temporal_history(session_id, repository, total_duration_s, -1):
                st.session_state['activity_management_notice'] = {'level': 'success', 'message': 'Modification annulee.'}
                st.rerun()
        if action_cols[1].button('Retablir', disabled=not can_redo, use_container_width=True, key=f'redo_seg_{session_id}'):
            if restore_temporal_history(session_id, repository, total_duration_s, 1):
                st.session_state['activity_management_notice'] = {'level': 'success', 'message': 'Modification retablie.'}
                st.rerun()
        if action_cols[2].button('Auto-segmenter', use_container_width=True, key=f'auto_seg_{session_id}'):
            if auto_generate_temporal_segments(session_id, repository, total_duration_s):
                st.session_state['activity_form_session_id'] = None
                st.rerun()
        if action_cols[3].button('Reinitialiser', use_container_width=True, key=f'reset_seg_{session_id}', disabled=not segments):
            if auto_generate_temporal_segments(session_id, repository, total_duration_s):
                st.session_state['activity_form_session_id'] = None
                st.rerun()

    hr_points = build_hr_points_for_component(hr_frame)
    component_value = FC_SEGMENT_EDITOR(
        hr_points=hr_points,
        segments=serialize_temporal_segments_for_component(segments),
        selected_segment_id=None if selected_segment_index is None else int(selected_segment_index),
        min_duration_s=10,
        default=None,
        key=f'fc_segment_editor_{session_id}',
    )

    if isinstance(component_value, dict):
        event_id = int(component_value.get('event_id', 0) or 0)
        last_event_id = int(st.session_state.get(temporal_component_event_key(session_id), 0) or 0)
        if event_id > last_event_id:
            st.session_state[temporal_component_event_key(session_id)] = event_id
            selected_segment_component_index = component_value.get('selected_segment_index')
            if selected_segment_component_index is not None:
                st.session_state[temporal_selected_segment_key(session_id)] = int(selected_segment_component_index)
                selected_segment_index = int(selected_segment_component_index)
            if component_value.get('event_type') == 'segment_selected':
                # The component interaction already causes one rerun.
                # Avoid forcing another one, which makes the page jump while annotating.
                pass
            elif component_value.get('event_type') == 'segments_updated':
                updated_segments = deserialize_temporal_segments_from_component(component_value.get('segments') or [])
                if persist_temporal_segments(session_id, updated_segments, total_duration_s, repository):
                    st.session_state['activity_management_notice'] = {'level': 'success', 'message': f'Segments temporels mis a jour pour {session_id}.'}

    segments = st.session_state.get(temporal_segments_key(session_id), [])
    selected_segment_index = st.session_state.get(temporal_selected_segment_key(session_id))
    active_segment = None
    if segments and selected_segment_index is not None and 0 <= int(selected_segment_index) < len(segments):
        active_segment = segments[int(selected_segment_index)]

    action_segment_key = f'popover_segment_type_{session_id}'
    action_segment_sync_key = f'popover_segment_sync_{session_id}'
    if active_segment is not None and (
        action_segment_key not in st.session_state
        or st.session_state.get(action_segment_sync_key) != int(selected_segment_index)
    ):
        st.session_state[action_segment_key] = str(active_segment.get('label') or 'technique')
        st.session_state[action_segment_sync_key] = int(selected_segment_index)

    with st.popover('Actions du segment', use_container_width=False):
        if active_segment is None:
            st.caption('Selectionne d abord un segment dans le graphe.')
        else:
            st.selectbox('Changer le type', options=TEMPORAL_SEGMENT_LABELS, key=action_segment_key)
            if st.button('Appliquer', key=f'popover_apply_type_{session_id}', use_container_width=True):
                if update_selected_temporal_segment(session_id, total_duration_s, repository, label=st.session_state.get(action_segment_key)):
                    st.rerun()
            st.markdown('---')
            new_segment_key = f'popover_new_segment_type_{session_id}'
            if new_segment_key not in st.session_state:
                st.session_state[new_segment_key] = 'randori_tw'
            st.selectbox('Type du nouveau segment', options=TEMPORAL_SEGMENT_LABELS, key=new_segment_key)
            insert_cols = st.columns(2)
            if insert_cols[0].button('Inserer a gauche', key=f'popover_insert_left_{session_id}', use_container_width=True):
                if split_selected_temporal_segment(session_id, st.session_state.get(new_segment_key, 'technique'), total_duration_s, repository, 'left'):
                    st.rerun()
            if insert_cols[1].button('Inserer a droite', key=f'popover_insert_right_{session_id}', use_container_width=True):
                if split_selected_temporal_segment(session_id, st.session_state.get(new_segment_key, 'technique'), total_duration_s, repository, 'right'):
                    st.rerun()
            st.markdown('---')
            confirm_key = f'confirm_delete_segment_{session_id}'
            st.checkbox('Confirmer la suppression', key=confirm_key)
            if st.button('Supprimer ce segment', key=f'popover_delete_{session_id}', use_container_width=True, disabled=not st.session_state.get(confirm_key, False)):
                if delete_temporal_segment(session_id, total_duration_s, repository):
                    st.session_state[confirm_key] = False
                    st.rerun()

    lower_cols = st.columns([3, 1])
    with lower_cols[0]:
        st.markdown("<div style='background:#f5f5f3;border-radius:12px;padding:16px;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.85rem;color:#667066;margin-bottom:0.75rem;font-weight:600;'>Segment selectionne</div>", unsafe_allow_html=True)
        if active_segment is None:
            st.info('Clique sur un segment du graphe pour le selectionner.')
        else:
            selected_type_key = f'selected_segment_type_compact_{session_id}'
            selected_type_sync_key = f'selected_segment_type_sync_{session_id}'
            if (
                selected_type_key not in st.session_state
                or st.session_state.get(selected_type_sync_key) != int(selected_segment_index)
            ):
                st.session_state[selected_type_key] = str(active_segment.get('label') or 'technique')
                st.session_state[selected_type_sync_key] = int(selected_segment_index)
            type_cols = st.columns([2.2, 1.0])
            type_cols[0].selectbox('Type', options=TEMPORAL_SEGMENT_LABELS, key=selected_type_key)
            if type_cols[1].button('Appliquer', key=f'compact_apply_type_{session_id}', use_container_width=True):
                if update_selected_temporal_segment(session_id, total_duration_s, repository, label=st.session_state.get(selected_type_key)):
                    st.rerun()
            metric_grid = st.columns(2)
            metric_grid[0].markdown(build_metric_card_html('Debut', format_offset_label(active_segment.get('start_offset_s'))), unsafe_allow_html=True)
            metric_grid[1].markdown(build_metric_card_html('Fin', format_offset_label(active_segment.get('end_offset_s'))), unsafe_allow_html=True)
            metric_grid = st.columns(2)
            metric_grid[0].markdown(build_metric_card_html('Duree', format_duration(active_segment.get('duration_s'))), unsafe_allow_html=True)
            metric_grid[1].markdown(build_metric_card_html('FC moyenne', compute_segment_fc_mean(hr_frame, active_segment)), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with lower_cols[1]:
        st.markdown("<div style='display:flex;flex-direction:column;gap:0.65rem;justify-content:center;height:100%;'>", unsafe_allow_html=True)
        if st.button('Valider segmentation', key=f'validate_fc_segments_{session_id}', type='primary', use_container_width=True, disabled=not segments):
            if persist_temporal_segments(session_id, segments, total_duration_s, repository):
                sync_result = repository.sync_segmentation_to_description(session_id)
                warnings = sync_result.get('warnings') or []
                randori_count = int(sync_result.get('randori_count') or 0)
                if warnings:
                    st.session_state['activity_management_notice'] = {
                        'level': 'warning',
                        'message': ' '.join(warnings),
                    }
                else:
                    st.session_state['activity_management_notice'] = {
                        'level': 'success',
                        'message': f'Description randoris mise a jour depuis la segmentation ({randori_count} randoris, durees et recuperations recalculees).',
                    }
                st.session_state['activity_form_session_id'] = None
                st.rerun()
        export_frame = repository.build_fc_phase_segments_export_frame(segments)
        export_csv = export_frame.to_csv(index=False) if not export_frame.empty else "segment_id,type,t_debut_s,t_fin_s,duree_s\n"
        st.download_button('Exporter CSV', data=export_csv, file_name=f'{session_id}_segments_fc.csv', mime='text/csv', use_container_width=True, key=f'export_fc_segments_{session_id}')
        st.markdown('</div>', unsafe_allow_html=True)

    summary_frame = repository.summarize_fc_phase_segments(st.session_state.get(temporal_segments_key(session_id), []))
    with st.expander(f'Resume des phases ({len(summary_frame) if not summary_frame.empty else 0} segments)', expanded=False):
        if summary_frame.empty:
            st.info('Aucun segment temporel a resumer.')
        else:
            summary_frame = summary_frame.copy()
            summary_frame['segment_idx'] = summary_frame['ordre'].astype(int)
            summary_frame['ordre_display'] = range(1, len(summary_frame) + 1)
            summary_frame['debut'] = summary_frame['debut_s'].apply(format_offset_label)
            summary_frame['fin'] = summary_frame['fin_s'].apply(format_offset_label)
            summary_frame['duree'] = summary_frame['duree_s'].apply(format_duration)
            active_segment_idx = int(selected_segment_index) if selected_segment_index is not None else None
            for row in summary_frame.itertuples(index=False):
                row_cols = st.columns([0.6, 1.8, 1.0, 1.0, 1.0])
                tone = 'background:rgba(33,72,52,0.06);border-radius:10px;' if active_segment_idx == int(row.segment_idx) else ''
                with row_cols[0]:
                    if st.button(str(row.ordre_display), key=f'select_summary_segment_{session_id}_{row.segment_idx}', use_container_width=True):
                        st.session_state[temporal_selected_segment_key(session_id)] = int(row.segment_idx)
                        st.rerun()
                with row_cols[1]:
                    st.markdown(build_segment_chip_html(str(row.phase)), unsafe_allow_html=True)
                row_cols[2].markdown(f"<div style='{tone}padding:0.4rem 0.6rem'>{row.debut}</div>", unsafe_allow_html=True)
                row_cols[3].markdown(f"<div style='{tone}padding:0.4rem 0.6rem'>{row.fin}</div>", unsafe_allow_html=True)
                row_cols[4].markdown(f"<div style='{tone}padding:0.4rem 0.6rem'>{row.duree}</div>", unsafe_allow_html=True)

    st.caption('Clic gauche = selectionner segment - Actions sur le segment = popover ci-dessus - Glisser bords = redimensionner - Ctrl+Z = annuler')


def initialize_form_state(session, repository: ProcessedSessionRepository) -> None:
    selected_session_key = "activity_form_session_id"
    form_signature_key = "activity_form_session_signature"
    session_id = session.session_id
    form_signature = (
        session_id,
        str(getattr(session, "updated_at", None) or ""),
        str(getattr(session, "activity_annotated_at", None) or ""),
        str(getattr(session, "temporally_annotated_at", None) or ""),
        len(session.judo_phases or []),
        len(session.judo_randori_blocks or []),
        len(session.fc_phase_segments or []),
    )
    if st.session_state.get(selected_session_key) == session_id and st.session_state.get(form_signature_key) == form_signature:
        return

    st.session_state[selected_session_key] = session_id
    st.session_state[form_signature_key] = form_signature
    st.session_state[f"annotation_input_{session_id}"] = session.annotation or session.session_id
    st.session_state[f"activity_family_{session_id}"] = session.activity_family or ""
    st.session_state[f"activity_label_{session_id}"] = session.activity_label or ""
    st.session_state[f"activity_notes_{session_id}"] = session.activity_notes or ""
    st.session_state[f"session_rpe_{session_id}"] = str(getattr(session, "session_rpe", None)) if getattr(session, "session_rpe", None) is not None else "NA"
    st.session_state[f"judo_session_type_{session_id}"] = session.judo_session_type or "technique"
    normalized_segments = repository.normalize_fc_phase_segments(
        [dict(item) for item in (session.fc_phase_segments or [])],
        max(float(session.duree_s or 0.0), 1.0),
    )
    st.session_state[temporal_segments_key(session_id)] = normalized_segments
    st.session_state[temporal_selected_segment_key(session_id)] = 0 if normalized_segments else None
    st.session_state[temporal_selected_point_key(session_id)] = None
    st.session_state[temporal_edit_mode_key(session_id)] = "inspection"
    st.session_state[temporal_move_anchor_key(session_id)] = "debut"
    st.session_state[temporal_last_graph_action_key(session_id)] = None
    st.session_state[temporal_component_event_key(session_id)] = 0
    reset_temporal_history(session_id, normalized_segments)

    phase_items: list[dict[str, Any]] = []
    max_phase_counter = 0
    for index, stored_phase in enumerate(session.judo_phases or [], start=1):
        phase_uid = stored_phase.get("phase_uid") or f"phase_{index}"
        phase_label = stored_phase.get("phase_label") or stored_phase.get("label") or "echauffement"
        phase_items.append({"phase_uid": phase_uid, "phase_label": phase_label})
        if isinstance(phase_uid, str) and phase_uid.startswith("phase_"):
            try:
                max_phase_counter = max(max_phase_counter, int(phase_uid.split("_")[-1]))
            except ValueError:
                pass
    st.session_state[f"phase_items_{session_id}"] = phase_items
    st.session_state[f"phase_counter_{session_id}"] = max(max_phase_counter, len(phase_items))

    randori_blocks_by_uid = {
        str(block.get("phase_uid")): block
        for block in (session.judo_randori_blocks or [])
        if isinstance(block, dict) and block.get("phase_uid")
    }
    randori_blocks_by_index = {
        block.get("phase_index"): block
        for block in (session.judo_randori_blocks or [])
        if isinstance(block, dict)
    }
    for phase_index, phase_item in enumerate(phase_items):
        phase_uid = phase_item["phase_uid"]
        phase_label = phase_item["phase_label"]
        st.session_state[f"phase_label_{session_id}_{phase_uid}"] = phase_label
        block = randori_blocks_by_uid.get(str(phase_uid)) or randori_blocks_by_index.get(phase_index, {})
        kind_default = str(block.get("randori_kind") or default_randori_kind(phase_label))
        count_default = str(block.get("randori_count") or "NA")
        rest_default = str(block.get("rest_between_randoris_min") or "NA")
        duration_default = str(block.get("randori_duration_min") or "4.0")
        st.session_state[f"randori_kind_{session_id}_{phase_uid}"] = kind_default
        st.session_state[f"randori_count_{session_id}_{phase_uid}"] = count_default
        st.session_state[f"randori_rest_{session_id}_{phase_uid}"] = rest_default
        st.session_state[f"randori_duration_{session_id}_{phase_uid}"] = duration_default
        randori_entries = block.get("randori_entries") or []
        entries_by_index: dict[int, dict[str, Any]] = {}
        for entry in randori_entries:
            if not isinstance(entry, dict):
                continue
            repetition_index = entry.get("repetition_index")
            if repetition_index is None:
                continue
            try:
                entries_by_index[int(repetition_index)] = entry
            except (TypeError, ValueError):
                continue
        try:
            count_int = int(block.get("randori_count")) if block.get("randori_count") not in (None, "", "NA") else 0
        except (TypeError, ValueError):
            count_int = 0
        for repetition_index in range(1, count_int + 1):
            entry = entries_by_index.get(repetition_index, {})
            entry_rpe = entry.get("rpe")
            st.session_state[f"randori_rpe_{session_id}_{phase_uid}_{repetition_index}"] = str(entry_rpe) if entry_rpe not in (None, "", "NA") else "NA"
            st.session_state[f"randori_comment_{session_id}_{phase_uid}_{repetition_index}"] = str(entry.get("comment") or "")
            entry_duration = entry.get("duration_min", block.get("randori_duration_min"))
            entry_recovery = entry.get("recovery_min", block.get("rest_between_randoris_min"))
            st.session_state[f"randori_duration_entry_{session_id}_{phase_uid}_{repetition_index}"] = float(entry_duration) if entry_duration not in (None, "", "NA") else 4.0
            st.session_state[f"randori_recovery_entry_{session_id}_{phase_uid}_{repetition_index}"] = float(entry_recovery) if entry_recovery not in (None, "", "NA") else 0.0


def append_phase(session_id: str, phase_label: str) -> None:
    phase_items_key = f"phase_items_{session_id}"
    counter_key = f"phase_counter_{session_id}"
    phase_items = list(st.session_state.get(phase_items_key, []))
    next_counter = int(st.session_state.get(counter_key, len(phase_items))) + 1
    phase_uid = f"phase_{next_counter}"
    phase_items.append({"phase_uid": phase_uid, "phase_label": phase_label})
    st.session_state[phase_items_key] = phase_items
    st.session_state[counter_key] = next_counter
    st.session_state[f"phase_label_{session_id}_{phase_uid}"] = phase_label
    st.session_state[f"randori_kind_{session_id}_{phase_uid}"] = default_randori_kind(phase_label)
    st.session_state[f"randori_count_{session_id}_{phase_uid}"] = "NA"
    st.session_state[f"randori_rest_{session_id}_{phase_uid}"] = "NA"
    st.session_state[f"randori_duration_{session_id}_{phase_uid}"] = "4.0"


def move_phase(session_id: str, phase_uid: str, direction: int) -> None:
    phase_items_key = f"phase_items_{session_id}"
    phase_items = list(st.session_state.get(phase_items_key, []))
    current_index = next((index for index, item in enumerate(phase_items) if item["phase_uid"] == phase_uid), None)
    if current_index is None:
        return
    target_index = current_index + direction
    if target_index < 0 or target_index >= len(phase_items):
        return
    phase_items[current_index], phase_items[target_index] = phase_items[target_index], phase_items[current_index]
    st.session_state[phase_items_key] = phase_items


def remove_phase(session_id: str, phase_uid: str) -> None:
    phase_items_key = f"phase_items_{session_id}"
    phase_items = [item for item in st.session_state.get(phase_items_key, []) if item["phase_uid"] != phase_uid]
    st.session_state[phase_items_key] = phase_items


def build_phase_payload(session_id: str, session: Any | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    phases: list[dict[str, Any]] = []
    randori_blocks: list[dict[str, Any]] = []
    preserve_sync = bool(getattr(session, "description_synced_from_segmentation", False))
    previous_blocks_by_uid: dict[str, dict[str, Any]] = {}
    if session is not None:
        for block in getattr(session, "judo_randori_blocks", None) or []:
            if isinstance(block, dict) and block.get("phase_uid") is not None:
                previous_blocks_by_uid[str(block.get("phase_uid"))] = block

    for phase_index, item in enumerate(st.session_state.get(f"phase_items_{session_id}", [])):
        phase_uid = item["phase_uid"]
        phase_label = st.session_state.get(f"phase_label_{session_id}_{phase_uid}", item["phase_label"])
        needs_randori_details = phase_is_randori(phase_label)
        phases.append(
            {
                "phase_index": phase_index,
                "phase_uid": phase_uid,
                "phase_label": phase_label,
                "phase_category": phase_category(phase_label),
                "needs_randori_details": needs_randori_details,
            }
        )
        if not needs_randori_details:
            continue

        randori_count = to_optional_number(st.session_state.get(f"randori_count_{session_id}_{phase_uid}"), integer=True)
        randori_kind = default_randori_kind(phase_label)
        randori_entries: list[dict[str, Any]] = []
        duration_values: list[float] = []
        recovery_values: list[float] = []
        if isinstance(randori_count, int) and randori_count > 0:
            for repetition_index in range(1, randori_count + 1):
                duration_value = float(st.session_state.get(f"randori_duration_entry_{session_id}_{phase_uid}_{repetition_index}", 4.0))
                recovery_value = float(st.session_state.get(f"randori_recovery_entry_{session_id}_{phase_uid}_{repetition_index}", 0.0))
                duration_values.append(duration_value)
                if repetition_index < randori_count and recovery_value > 0:
                    recovery_values.append(recovery_value)
                randori_entries.append(
                    {
                        "repetition_index": repetition_index,
                        "rpe": to_optional_int(st.session_state.get(f"randori_rpe_{session_id}_{phase_uid}_{repetition_index}")),
                        "comment": to_optional_text(st.session_state.get(f"randori_comment_{session_id}_{phase_uid}_{repetition_index}")),
                        "duration_min": duration_value,
                        "recovery_min": recovery_value if repetition_index < randori_count else None,
                    }
                )

        block_duration = round(sum(duration_values) / len(duration_values), 2) if duration_values else to_optional_number(st.session_state.get(f"randori_duration_{session_id}_{phase_uid}"), integer=False)
        block_recovery = round(sum(recovery_values) / len(recovery_values), 2) if recovery_values else to_optional_number(st.session_state.get(f"randori_rest_{session_id}_{phase_uid}"), integer=False)
        randori_blocks.append(
            {
                "phase_index": phase_index,
                "phase_uid": phase_uid,
                "randori_kind": randori_kind,
                "randori_count": randori_count,
                "randori_duration_min": block_duration,
                "rest_between_randoris_min": block_recovery,
                "randori_entries": randori_entries,
            }
        )

        if preserve_sync:
            previous_block = previous_blocks_by_uid.get(str(phase_uid), {})
            previous_entries = {
                int(entry.get("repetition_index")): entry
                for entry in (previous_block.get("randori_entries") or [])
                if isinstance(entry, dict) and entry.get("repetition_index") is not None
            }
            previous_count = previous_block.get("randori_count")
            current_count = randori_count if isinstance(randori_count, int) else 0
            if (previous_count or 0) != current_count:
                preserve_sync = False
            for entry in randori_entries:
                previous_entry = previous_entries.get(int(entry.get("repetition_index")), {})
                if previous_entry.get("duration_min") != entry.get("duration_min"):
                    preserve_sync = False
                if previous_entry.get("recovery_min") != entry.get("recovery_min"):
                    preserve_sync = False

    return phases, randori_blocks, preserve_sync


def build_randori_description_groups(session_id: str, phase_rows: list[tuple[int, str, str]]) -> list[dict[str, Any]]:
    groups_map: dict[str, dict[str, Any]] = {}
    group_order: list[str] = []
    for _, phase_uid, current_phase_label in phase_rows:
        if not phase_is_randori(current_phase_label):
            continue
        phase_type = str(current_phase_label).strip().lower()
        selected_count = st.session_state.get(f'randori_count_{session_id}_{phase_uid}', 'NA')
        try:
            repetition_count = int(selected_count) if selected_count not in (None, '', 'NA') else 0
        except (TypeError, ValueError):
            repetition_count = 0
        if repetition_count <= 0:
            continue
        if phase_type not in groups_map:
            groups_map[phase_type] = {'phase_label': phase_type, 'items': []}
            group_order.append(phase_type)
        for repetition_index in range(1, repetition_count + 1):
            rpe_key = f'randori_rpe_{session_id}_{phase_uid}_{repetition_index}'
            comment_key = f'randori_comment_{session_id}_{phase_uid}_{repetition_index}'
            duration_key = f'randori_duration_entry_{session_id}_{phase_uid}_{repetition_index}'
            recovery_key = f'randori_recovery_entry_{session_id}_{phase_uid}_{repetition_index}'
            if rpe_key not in st.session_state:
                st.session_state[rpe_key] = 'NA'
            if comment_key not in st.session_state:
                st.session_state[comment_key] = ''
            if duration_key not in st.session_state:
                st.session_state[duration_key] = 4.0
            if recovery_key not in st.session_state:
                st.session_state[recovery_key] = 0.0
            groups_map[phase_type]['items'].append({'phase_uid': phase_uid, 'phase_label': phase_type, 'repetition_index': repetition_index})
    return [groups_map[key] for key in group_order]


def get_selected_session_id_from_calendar(calendar_state: Any) -> str | None:
    if not isinstance(calendar_state, dict):
        return None
    event_click = calendar_state.get("eventClick")
    if isinstance(event_click, dict):
        event = event_click.get("event") if isinstance(event_click.get("event"), dict) else event_click
        if isinstance(event, dict):
            return (
                event.get("id")
                or ((event.get("extendedProps") or {}).get("session_id") if isinstance(event.get("extendedProps"), dict) else None)
            )
    return None


def render_hr_chart(hr_frame: pd.DataFrame) -> None:
    chart_frame = build_hr_chart_frame(hr_frame)
    if chart_frame.empty or chart_frame["bpm"].dropna().empty:
        st.info("Aucune donnee FC brute disponible pour cette seance.")
        return
    chart = (
        alt.Chart(chart_frame.dropna(subset=["bpm"]))
        .mark_line(strokeWidth=2.1, color="#2f6f4e")
        .encode(
            x=alt.X("t_min:Q", title="Temps (min)"),
            y=alt.Y("bpm:Q", title="FC brute (bpm)"),
            detail="line_group:N",
            tooltip=["t_min:Q", "bpm:Q"],
        )
    )
    st.altair_chart(chart.properties(height=280), use_container_width=True)


def render_session_summary(session, hr_frame: pd.DataFrame, repository: ProcessedSessionRepository) -> None:
    start_dt = repository.get_session_start(session)
    end_dt = repository.get_session_end(session)
    hr_valid = hr_frame.loc[hr_frame["bpm"].fillna(0) > 0].copy() if not hr_frame.empty and "bpm" in hr_frame.columns else pd.DataFrame()
    fc_min = int(hr_valid["bpm"].min()) if not hr_valid.empty else session.bpm_min
    fc_max = int(hr_valid["bpm"].max()) if not hr_valid.empty else session.bpm_max

    cols = st.columns(6)
    cols[0].metric("Duree", format_duration(session.duree_s))
    cols[1].metric("FC min brute", f"{fc_min} bpm")
    cols[2].metric("FC max brute", f"{fc_max} bpm")
    cols[3].metric("Heure debut", format_datetime_label(start_dt))
    cols[4].metric("Heure fin", format_datetime_label(end_dt))
    cols[5].metric("RPE seance", str(session.session_rpe) if getattr(session, "session_rpe", None) is not None else "-")


def render_status_pills(session) -> None:
    annotated_html = "Annotee" if session.is_activity_annotated else "Non annotee"
    archive_html = "Archivee" if session.is_archived else "Active"
    annotated_class = "status-pill" if session.is_activity_annotated else "status-pill warn"
    archive_class = "status-pill muted" if session.is_archived else "status-pill"
    pills = [
        f'<span class="{annotated_class}">{annotated_html}</span>',
        f'<span class="{archive_class}">{archive_html}</span>',
        f'<span class="status-pill muted">{session.activity_family or "famille non renseignee"}</span>',
        f'<span class="status-pill muted">{session.activity_label or "activite non renseignee"}</span>',
    ]
    if session.activity_family == "judo" and session.judo_session_type == "randoris":
        temporal_class = "status-pill" if session.is_temporally_annotated else "status-pill warn"
        temporal_label = "Annotation temporelle oui" if session.is_temporally_annotated else "Annotation temporelle non"
        pills.append(f'<span class="{temporal_class}">{temporal_label}</span>')
    st.markdown(f'<div class="status-row">{"".join(pills)}</div>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Gestion activites", layout="wide", initial_sidebar_state="expanded")
    inject_styles()
    render_notice()
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Gestion des activites</div>
            <div class="hero-subtitle">Calendrier des seances, annotation metier, archivage logique et description detaillee des seances de judo.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    activity_options = load_reference_lines(ACTIVITY_REFERENCE_PATH)
    judo_phase_options = load_reference_lines(JUDO_PHASE_REFERENCE_PATH)

    with st.sidebar:
        render_section_label("Parametres")
        output_dir = st.text_input("Dossier data", value=DEFAULT_OUTPUT_DIR)
        family_filter = st.selectbox("Famille", options=["toutes", "judo", "prepa", "autres"], index=0)
        annotated_only = st.toggle("Annotees seulement", value=False)
        show_archived = st.toggle("Afficher archivees", value=False)

    repository = ProcessedSessionRepository(output_dir)
    visible_sessions = repository.list_sessions(include_archived=show_archived)
    if family_filter != "toutes":
        visible_sessions = [session for session in visible_sessions if session.activity_family == family_filter]
    if annotated_only:
        visible_sessions = [session for session in visible_sessions if session.is_activity_annotated]
    visible_sessions = sorted(visible_sessions, key=lambda session: (session.date, session.heure_debut, session.session_id))

    if not visible_sessions:
        st.info("Aucune activite ne correspond aux filtres actuels.")
        st.stop()

    selected_session_id = st.session_state.get("selected_activity_id")
    if selected_session_id not in {session.session_id for session in visible_sessions}:
        selected_session_id = visible_sessions[-1].session_id
        st.session_state["selected_activity_id"] = selected_session_id

    events = highlight_selected_event(
        repository.build_calendar_events(
            include_archived=show_archived,
            family_filter=family_filter,
            annotated_only=annotated_only,
        ),
        selected_session_id,
    )

    render_section_label("Calendrier")
    if calendar is None:
        st.warning("Le module `streamlit-calendar` n'est pas installe localement. La dependance est ajoutee dans `requirements.txt`, et la page bascule temporairement sur une selection simple.")
        fallback_options = {
            f"{session.annotation or session.session_id} | {session.date} {session.heure_debut}": session.session_id
            for session in visible_sessions
        }
        selected_label = st.selectbox("Activite", options=list(fallback_options.keys()), index=len(fallback_options) - 1)
        st.session_state["selected_activity_id"] = fallback_options[selected_label]
    else:
        calendar_options = {
            "initialView": "dayGridMonth",
            "headerToolbar": {
                "left": "today prev,next",
                "center": "title",
                "right": "dayGridMonth,timeGridWeek,timeGridDay",
            },
            "height": 680,
            "locale": "fr",
            "editable": False,
            "selectable": False,
            "eventDisplay": "block",
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
            key="activity_calendar",
        )
        clicked_session_id = get_selected_session_id_from_calendar(calendar_state)
        if clicked_session_id and clicked_session_id != st.session_state.get("selected_activity_id"):
            st.session_state["selected_activity_id"] = clicked_session_id
            st.session_state["activity_form_session_id"] = None
            st.rerun()

    selected_session_id = st.session_state.get("selected_activity_id", selected_session_id)
    session = repository.get_sessions([selected_session_id], include_archived=True)[0]
    initialize_form_state(session, repository)
    _, _, hr_frame = repository.load_session_data(selected_session_id)

    st.markdown('<div class="detail-card">', unsafe_allow_html=True)
    start_dt = repository.get_session_start(session)
    end_dt = repository.get_session_end(session)
    hr_valid = hr_frame.loc[hr_frame["bpm"].fillna(0) > 0].copy() if not hr_frame.empty and "bpm" in hr_frame.columns else pd.DataFrame()
    fc_min = int(hr_valid["bpm"].min()) if not hr_valid.empty else session.bpm_min
    fc_max = int(hr_valid["bpm"].max()) if not hr_valid.empty else session.bpm_max
    title_cols = st.columns([3.2, 1.4])
    with title_cols[0]:
        st.markdown(f"<div style='font-size:22px;font-weight:700;color:#173427;'>{session.annotation or session.session_id}</div>", unsafe_allow_html=True)
        subtitle = f"{session.date} - {session.session_id} - {terminal_category_label(session.activity_label)}"
        st.markdown(f"<div style='font-size:0.86rem;color:#667066;margin-top:0.18rem;'>{subtitle}</div>", unsafe_allow_html=True)
    with title_cols[1]:
        badges = [
            build_badge_html('Annotee' if session.is_activity_annotated else 'Non annotee', tone='success' if session.is_activity_annotated else 'warn'),
            build_badge_html('Active' if not session.is_archived else 'Archivee', tone='default' if not session.is_archived else 'muted'),
        ]
        st.markdown(f"<div style='display:flex;gap:0.5rem;justify-content:flex-end;flex-wrap:wrap;'>{''.join(badges)}</div>", unsafe_allow_html=True)
    metric_cols = st.columns(6)
    metric_cols[0].markdown(build_metric_card_html('Duree', format_duration(session.duree_s)), unsafe_allow_html=True)
    metric_cols[1].markdown(build_metric_card_html('FC min', str(fc_min), 'bpm'), unsafe_allow_html=True)
    metric_cols[2].markdown(build_metric_card_html('FC max', str(fc_max), 'bpm'), unsafe_allow_html=True)
    metric_cols[3].markdown(build_metric_card_html('Debut', format_datetime_label(start_dt)), unsafe_allow_html=True)
    metric_cols[4].markdown(build_metric_card_html('Fin', format_datetime_label(end_dt)), unsafe_allow_html=True)
    metric_cols[5].markdown(build_metric_card_html('RPE seance', str(session.session_rpe) if getattr(session, 'session_rpe', None) is not None else '-'), unsafe_allow_html=True)

    session_id = session.session_id
    annotation_key = f"annotation_input_{session_id}"
    family_key = f"activity_family_{session_id}"
    label_key = f"activity_label_{session_id}"
    notes_key = f"activity_notes_{session_id}"
    session_rpe_key = f"session_rpe_{session_id}"
    judo_type_key = f"judo_session_type_{session_id}"
    selected_family = str(st.session_state.get(family_key) or session.activity_family or '')
    current_label = str(st.session_state.get(label_key) or session.activity_label or '')

    detail_toggle_key = f'detail_flow_toggle_{session_id}'
    detail_enabled_key = f'detail_flow_enabled_{session_id}'
    if detail_enabled_key not in st.session_state:
        st.session_state[detail_enabled_key] = bool(st.session_state.get(f'phase_items_{session_id}', []))
    if detail_toggle_key not in st.session_state:
        st.session_state[detail_toggle_key] = bool(st.session_state.get(detail_enabled_key, False))

    main_cols = st.columns([1, 1])
    with main_cols[0]:
        st.markdown("<div class='detail-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:1rem;font-weight:650;color:#173427;margin-bottom:0.9rem;'>Informations generales</div>", unsafe_allow_html=True)
        st.text_input('Nom de l activite', key=annotation_key)
        general_cols = st.columns(2)
        with general_cols[0]:
            st.selectbox('Famille d activite', options=['', 'judo', 'prepa', 'autres'], key=family_key)
        with general_cols[1]:
            if selected_family == 'judo':
                st.selectbox('Type de seance', options=['technique', 'randoris'], key=judo_type_key)
                selected_judo_type = st.session_state.get(judo_type_key)
                if selected_judo_type == 'technique':
                    st.session_state[label_key] = 'judo > technique'
                else:
                    randori_options = [option for option in activity_options if option.startswith('judo > randoris >')]
                    if current_label not in randori_options:
                        st.session_state[label_key] = randori_options[0] if randori_options else ''
                    st.selectbox('Sous-type', options=randori_options, key=label_key)
            elif selected_family == 'prepa':
                st.session_state[judo_type_key] = 'technique'
                prepa_options = [option for option in activity_options if option.startswith('prepa >')]
                if current_label not in prepa_options:
                    st.session_state[label_key] = prepa_options[0] if prepa_options else ''
                st.selectbox('Type de seance', options=prepa_options, key=label_key)
            elif selected_family == 'autres':
                st.session_state[judo_type_key] = 'technique'
                st.session_state[label_key] = 'autres'
                st.text_input('Categorie', value='autres', disabled=True)
            else:
                st.session_state[judo_type_key] = 'technique'
                st.session_state[label_key] = ''
                st.text_input('Type de seance', value='-', disabled=True)
        st.text_area('Notes generales', key=notes_key, height=110)
        general_footer_cols = st.columns(2)
        with general_footer_cols[0]:
            st.selectbox('RPE global', options=RPE_OPTIONS, key=session_rpe_key)
        with general_footer_cols[1]:
            st.text_input('Categorie', value=terminal_category_label(st.session_state.get(label_key)), disabled=True)
        st.markdown('</div>', unsafe_allow_html=True)

    phase_items = st.session_state.get(f'phase_items_{session_id}', [])
    phase_rows: list[tuple[int, str, str]] = []
    has_randori_phase = False
    randori_details_unlocked = False
    with main_cols[1]:
        st.markdown("<div class='detail-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:1rem;font-weight:650;color:#173427;margin-bottom:0.9rem;'>Deroule de la seance</div>", unsafe_allow_html=True)
        if st.session_state.get(family_key) != 'judo' or st.session_state.get(judo_type_key) != 'randoris':
            st.info('Aucun deroule detaille pour cette seance.')
        elif not phase_items and not st.session_state.get(detail_enabled_key):
            st.info('Aucun deroule detaille.')
            if st.button('Ajouter un deroule', key=f'activate_flow_{session_id}'):
                st.session_state[detail_enabled_key] = True
                st.rerun()
        else:
            st.toggle('Detailler le deroule', key=detail_toggle_key)
            st.session_state[detail_enabled_key] = bool(st.session_state.get(detail_toggle_key, False))
            add_phase_cols = st.columns([3, 1])
            new_phase_key = f'new_phase_{session_id}'
            if new_phase_key not in st.session_state:
                st.session_state[new_phase_key] = judo_phase_options[0] if judo_phase_options else 'echauffement'
            add_phase_cols[0].selectbox('Ajouter une phase', options=judo_phase_options or ['echauffement'], key=new_phase_key)
            if add_phase_cols[1].button('+ Phase', key=f'add_phase_button_{session_id}', use_container_width=True):
                append_phase(session_id, st.session_state.get(new_phase_key, 'echauffement'))
                st.session_state[detail_enabled_key] = True
                st.rerun()

            current_labels = current_phase_labels(session_id)
            persisted_labels = saved_phase_labels(session)
            has_randori_phase = any(phase_is_randori(label) for label in current_labels)
            randori_details_unlocked = bool(persisted_labels) and current_labels == persisted_labels
            if phase_items:
                pills_markup = ''.join(build_segment_chip_html(st.session_state.get(f'phase_label_{session_id}_{item["phase_uid"]}', item['phase_label'])) for item in phase_items)
                st.markdown(f"<div style='display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.6rem;'>{pills_markup}</div>", unsafe_allow_html=True)
                st.caption('Cliquer pour modifier - X pour retirer')
            for index, phase_item in enumerate(phase_items):
                phase_uid = phase_item['phase_uid']
                phase_label_key = f'phase_label_{session_id}_{phase_uid}'
                row_cols = st.columns([2.6, 0.7, 0.7, 0.7])
                row_cols[0].selectbox('Phase', options=judo_phase_options or ['echauffement'], key=phase_label_key, label_visibility='collapsed')
                if row_cols[1].button('Up', key=f'move_up_{session_id}_{phase_uid}', use_container_width=True):
                    move_phase(session_id, phase_uid, -1)
                    st.rerun()
                if row_cols[2].button('Down', key=f'move_down_{session_id}_{phase_uid}', use_container_width=True):
                    move_phase(session_id, phase_uid, 1)
                    st.rerun()
                if row_cols[3].button('X', key=f'remove_phase_{session_id}_{phase_uid}', use_container_width=True):
                    remove_phase(session_id, phase_uid)
                    st.rerun()
                phase_rows.append((index, phase_uid, st.session_state.get(phase_label_key, phase_item['phase_label'])))

            if has_randori_phase and not randori_details_unlocked:
                st.info('Commence par enregistrer la structure de la seance pour debloquer les precisions randoris.')
            elif has_randori_phase:
                st.markdown("<div style='margin-top:0.8rem;font-size:0.9rem;font-weight:600;color:#173427;'>Details phases randoris</div>", unsafe_allow_html=True)
                for index, phase_uid, current_phase_label in phase_rows:
                    if not phase_is_randori(current_phase_label):
                        continue
                    count_key = f'randori_count_{session_id}_{phase_uid}'
                    detail_cols = st.columns([1.5, 1.0])
                    detail_cols[0].text_input(f'Type phase {index + 1}', value=str(current_phase_label).replace('_', ' '), disabled=True)
                    detail_cols[1].selectbox(f'Nombre phase {index + 1}', options=RANDORI_COUNT_OPTIONS, key=count_key)
        st.markdown('</div>', unsafe_allow_html=True)

    if has_randori_phase and randori_details_unlocked:
        randori_groups = build_randori_description_groups(session_id, phase_rows)
        for group in randori_groups:
            randori_items = group['items']
            if not randori_items:
                continue
            rated_values = [
                int(st.session_state.get(f"randori_rpe_{session_id}_{item['phase_uid']}_{item['repetition_index']}"))
                for item in randori_items
                if st.session_state.get(f"randori_rpe_{session_id}_{item['phase_uid']}_{item['repetition_index']}") not in (None, '', 'NA')
            ]
            mean_rpe = round(sum(rated_values) / len(rated_values), 1) if rated_values else None
            phase_title = 'TW' if group['phase_label'] == 'randori_tw' else 'NW'
            st.markdown("<div class='detail-card'>", unsafe_allow_html=True)
            header_cols = st.columns([2, 1])
            header_cols[0].markdown(f"<div style='font-size:1rem;font-weight:650;color:#173427;'>Description randoris {phase_title}</div>", unsafe_allow_html=True)
            header_cols[1].markdown(f"<div style='text-align:right;color:#5b675e;font-size:0.9rem;margin-top:0.2rem;'>{len(randori_items)} randoris - RPE moyen : {mean_rpe if mean_rpe is not None else '-'}</div>", unsafe_allow_html=True)
            if getattr(session, 'description_synced_from_segmentation', False):
                st.markdown(build_badge_html('Durees et recuperations issues de la segmentation manuelle', tone='muted'), unsafe_allow_html=True)
            for row_index, row_items in enumerate(chunked(randori_items, 8)):
                row_cols = st.columns(len(row_items))
                row_offset = row_index * 8
                for display_index, (col, item) in enumerate(zip(row_cols, row_items), start=row_offset + 1):
                    phase_uid = item['phase_uid']
                    repetition_index = item['repetition_index']
                    rpe_key = f'randori_rpe_{session_id}_{phase_uid}_{repetition_index}'
                    comment_key = f'randori_comment_{session_id}_{phase_uid}_{repetition_index}'
                    duration_key = f'randori_duration_entry_{session_id}_{phase_uid}_{repetition_index}'
                    recovery_key = f'randori_recovery_entry_{session_id}_{phase_uid}_{repetition_index}'
                    raw_rpe = st.session_state.get(rpe_key, 'NA')
                    numeric_rpe = int(raw_rpe) if raw_rpe not in (None, '', 'NA') else 0
                    if numeric_rpe <= 5:
                        bar_color = '#1D9E75'
                    elif numeric_rpe <= 7:
                        bar_color = '#BA7517'
                    else:
                        bar_color = '#D85A30' if group['phase_label'] == 'randori_tw' else '#D4537E'
                    bar_height = max(12, numeric_rpe * 10)
                    with col:
                        st.markdown(f"<div style='font-weight:700;margin-bottom:0.35rem;'>R{display_index}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='height:112px;display:flex;align-items:flex-end;justify-content:center;background:#f7f7f5;border-radius:10px;margin-bottom:0.6rem;'><div style='width:42px;height:{bar_height}%;min-height:24px;background:{bar_color};border-radius:10px 10px 6px 6px;color:white;display:flex;align-items:center;justify-content:center;font-weight:700;'>{numeric_rpe if numeric_rpe else '-'}</div></div>", unsafe_allow_html=True)
                        st.selectbox('RPE', options=RPE_OPTIONS, key=rpe_key, label_visibility='collapsed')
                        st.number_input('Duree (min)', min_value=0.0, step=0.5, key=duration_key)
                        st.number_input('Recup (min)', min_value=0.0, step=0.5, key=recovery_key)
                        st.text_input('Commentaire', key=comment_key, label_visibility='collapsed', placeholder='Commentaire rapide')
            st.markdown('</div>', unsafe_allow_html=True)

    action_cols = st.columns([1, 1, 2])
    if action_cols[0].button("Enregistrer", type="primary", use_container_width=True):
        payload: dict[str, Any] = {
            "annotation": st.session_state.get(annotation_key),
            "activity_family": st.session_state.get(family_key) or None,
            "activity_label": st.session_state.get(label_key) or None,
            "activity_notes": st.session_state.get(notes_key) or None,
            "session_rpe": to_optional_int(st.session_state.get(session_rpe_key)),
        }
        if st.session_state.get(family_key) == "judo":
            payload["judo_session_type"] = st.session_state.get(judo_type_key)
            if st.session_state.get(judo_type_key) == "randoris":
                judo_phases, judo_randori_blocks, description_sync_preserved = build_phase_payload(session_id, session)
                payload["judo_phases"] = judo_phases
                payload["judo_randori_blocks"] = judo_randori_blocks
                payload["description_synced_from_segmentation"] = description_sync_preserved
                payload["description_synced_at"] = getattr(session, "description_synced_at", None) if description_sync_preserved else None
        repository.update_activity_metadata(session_id, payload)
        st.session_state["activity_management_notice"] = {
            "level": "success",
            "message": f"Metadonnees activite enregistrees pour {session_id}.",
        }
        st.session_state["activity_form_session_id"] = None
        st.rerun()

    if not session.is_archived:
        if action_cols[1].button("Archiver", use_container_width=True):
            repository.archive_session(session_id)
            st.session_state["activity_management_notice"] = {
                "level": "success",
                "message": f"Activite archivee : {session.annotation or session_id}.",
            }
            st.session_state["activity_form_session_id"] = None
            st.rerun()
    else:
        if action_cols[1].button("Restaurer", use_container_width=True):
            repository.restore_session(session_id)
            st.session_state["activity_management_notice"] = {
                "level": "success",
                "message": f"Activite restauree : {session.annotation or session_id}.",
            }
            st.session_state["activity_form_session_id"] = None
            st.rerun()

    current_segments = st.session_state.get(temporal_segments_key(session_id), [])
    segmentation_valid = (
        st.session_state.get(family_key) == 'judo'
        and st.session_state.get(judo_type_key) == 'randoris'
        and repository.validate_fc_phase_segments(current_segments, float(session.duree_s or 0.0))
        and bool(current_segments)
    )
    if st.session_state.get(family_key) == 'judo' and st.session_state.get(judo_type_key) == 'randoris':
        segmentation_badge = build_badge_html(
            'Segmentation : validee' if segmentation_valid else 'Segmentation : non segmentee',
            tone='success' if segmentation_valid else 'muted',
        )
    else:
        segmentation_badge = build_badge_html('Segmentation : non applicable', tone='muted')
    action_cols[2].markdown(
        f"<div style='display:flex;justify-content:flex-end;align-items:center;height:100%;'>{segmentation_badge}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if session.activity_family == "judo" and session.judo_session_type == "randoris":
        render_temporal_annotation_module(session, hr_frame, repository)
    else:
        render_section_label("FC brute")
        render_hr_chart(hr_frame)
    with st.expander("Apercu tabulaire FC brute"):
        st.dataframe(hr_frame.head(50), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()





