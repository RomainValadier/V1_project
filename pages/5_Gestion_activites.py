from __future__ import annotations

import os
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
RANDORI_KIND_OPTIONS = ["TW", "NW pure", "mixtes (NW + TW)", "libres"]
RANDORI_DURATION_OPTIONS = [f"{value:.1f}" for value in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]]
RANDORI_COUNT_OPTIONS = ["NA"] + [str(value) for value in range(1, 31)]
RANDORI_REST_OPTIONS = ["NA"] + [f"{value:.1f}" for value in [index / 2 for index in range(1, 21)]]
RANDORI_PHASE_LABELS = {"randoris TW", "randoris NW pure", "randoris libres"}
TEMPORAL_SEGMENT_LABELS = ["echauffement", "technique", "randori", "recuperation", "retour_calme", "autre"]
TEMPORAL_SEGMENT_COLORS = {
    "echauffement": "rgba(234, 179, 8, 0.18)",
    "technique": "rgba(59, 130, 246, 0.18)",
    "randori": "rgba(220, 38, 38, 0.20)",
    "recuperation": "rgba(22, 163, 74, 0.18)",
    "retour_calme": "rgba(139, 92, 246, 0.18)",
    "autre": "rgba(107, 114, 128, 0.18)",
}
TEMPORAL_SEGMENT_HEX_COLORS = {
    "echauffement": "#f59e0b",
    "technique": "#3b82f6",
    "randori": "#ef4444",
    "recuperation": "#22c55e",
    "retour_calme": "#8b5cf6",
    "autre": "#6b7280",
}
SELECTED_EVENT_BACKGROUND = "#f59e0b"
SELECTED_EVENT_BORDER = "#b45309"
SELECTED_EVENT_TEXT = "#1f2937"
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_label(label: str) -> None:
    st.markdown(f'<div class="section-chip">{label}</div>', unsafe_allow_html=True)


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
    return phase_label in RANDORI_PHASE_LABELS


def phase_category(phase_label: str) -> str:
    return "randoris" if phase_is_randori(phase_label) else phase_label


def default_randori_kind(phase_label: str) -> str:
    if phase_label == "randoris TW":
        return "TW"
    if phase_label == "randoris NW pure":
        return "NW pure"
    if phase_label == "randoris libres":
        return "libres"
    return "mixtes (NW + TW)"


def to_optional_number(raw_value: str | None, integer: bool = False) -> str | int | float:
    if raw_value in (None, "", "NA"):
        return "NA"
    return int(raw_value) if integer else float(raw_value)


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
        label = str(segment.get("label") or "autre")
        serialized.append(
            {
                "id": int(segment.get("segment_index", index)),
                "type": label,
                "t_debut_s": round(float(segment.get("start_offset_s", 0.0)), 3),
                "t_fin_s": round(float(segment.get("end_offset_s", 0.0)), 3),
                "color": TEMPORAL_SEGMENT_HEX_COLORS.get(label, TEMPORAL_SEGMENT_HEX_COLORS["autre"]),
            }
        )
    return serialized


def deserialize_temporal_segments_from_component(raw_segments: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    deserialized: list[dict[str, Any]] = []
    for index, segment in enumerate(raw_segments or []):
        label = str(segment.get("type") or segment.get("label") or "autre")
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
        color = TEMPORAL_SEGMENT_COLORS.get(segment.get("label"), "rgba(99, 102, 241, 0.14)")
        border_color = "#b45309" if is_selected else "rgba(30,41,59,0.28)"
        fig.add_vrect(
            x0=float(segment.get("start_offset_s", 0.0)),
            x1=float(segment.get("end_offset_s", 0.0)),
            fillcolor=color,
            opacity=0.96 if is_selected else 0.58,
            layer="below",
            line_width=4 if is_selected else 1.5,
            line_color=border_color,
            annotation_text=str(segment.get("label", "segment")),
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


def render_temporal_annotation_module(session, hr_frame: pd.DataFrame, repository: ProcessedSessionRepository) -> None:
    session_id = session.session_id
    total_duration_s = max(float(session.duree_s or 0.0), float(hr_frame["t_offset_ms"].max()) / 1000.0 if not hr_frame.empty else 0.0)
    segments = st.session_state.get(temporal_segments_key(session_id), [])
    selected_segment_index = st.session_state.get(temporal_selected_segment_key(session_id))

    render_section_label("Annotations temporelles FC")
    top_cols = st.columns([1.0, 1.0, 1.1, 1.0])
    if top_cols[0].button("Auto-generer", key=f"generate_fc_segments_{session_id}", use_container_width=True):
        try:
            generated_segments = repository.generate_default_fc_phase_segments(session_id)
            if persist_temporal_segments(session_id, generated_segments, total_duration_s, repository):
                st.session_state[temporal_component_event_key(session_id)] = 0
                st.session_state["activity_management_notice"] = {"level": "success", "message": f"Annotations temporelles generees pour {session_id}."}
                st.rerun()
        except Exception as exc:
            st.session_state["activity_management_notice"] = {"level": "warning", "message": str(exc)}
            st.rerun()
    if top_cols[1].button("Reinitialiser", key=f"reset_fc_segments_{session_id}", use_container_width=True):
        try:
            generated_segments = repository.generate_default_fc_phase_segments(session_id)
            if persist_temporal_segments(session_id, generated_segments, total_duration_s, repository):
                st.session_state[temporal_component_event_key(session_id)] = 0
                st.session_state["activity_management_notice"] = {"level": "success", "message": f"Annotations temporelles reinitialisees pour {session_id}."}
                st.rerun()
        except Exception as exc:
            st.session_state["activity_management_notice"] = {"level": "warning", "message": str(exc)}
            st.rerun()
    top_cols[2].metric("Statut", temporal_status_label(segments))
    top_cols[3].metric("Segments", str(len(segments)))

    st.caption(
        "Interaction : clic gauche dans un segment pour le selectionner, puis glisse un bord interne pour redimensionner la seance. Les segments restent contigus et aucune phase ne peut descendre sous 10 secondes."
    )


    hr_points = build_hr_points_for_component(hr_frame)
    component_value = FC_SEGMENT_EDITOR(
        hr_points=hr_points,
        segments=serialize_temporal_segments_for_component(segments),
        selected_segment_id=None if selected_segment_index is None else int(selected_segment_index),
        min_duration_s=10,
        default=None,
        key=f"fc_segment_editor_{session_id}",
    )

    if isinstance(component_value, dict):
        event_id = int(component_value.get("event_id", 0) or 0)
        last_event_id = int(st.session_state.get(temporal_component_event_key(session_id), 0) or 0)
        if event_id > last_event_id:
            st.session_state[temporal_component_event_key(session_id)] = event_id
            selected_segment_id = component_value.get("selected_segment_id")
            previous_selected_segment_index = st.session_state.get(temporal_selected_segment_key(session_id))
            if selected_segment_id is not None:
                st.session_state[temporal_selected_segment_key(session_id)] = int(selected_segment_id)
                selected_segment_index = int(selected_segment_id)
            if component_value.get("event_type") == "segment_selected":
                if selected_segment_id is not None and previous_selected_segment_index != int(selected_segment_id):
                    st.rerun()
            elif component_value.get("event_type") == "segments_updated":
                raw_segments = component_value.get("segments") or []
                updated_segments = deserialize_temporal_segments_from_component(raw_segments)
                if persist_temporal_segments(
                    session_id,
                    updated_segments,
                    total_duration_s,
                    repository,
                ):
                    segments = st.session_state.get(temporal_segments_key(session_id), [])
                    st.session_state["activity_management_notice"] = {"level": "success", "message": f"Segments temporels mis a jour pour {session_id}."}
                    st.rerun()
            else:
                segments = st.session_state.get(temporal_segments_key(session_id), [])

    st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)
    selected_segment_index = st.session_state.get(temporal_selected_segment_key(session_id))
    active_segment = None
    if segments and selected_segment_index is not None and 0 <= int(selected_segment_index) < len(segments):
        active_segment = segments[int(selected_segment_index)]

    action_cols = st.columns([1.65, 1.0])
    with action_cols[0]:
        render_section_label("Segment actif")
        if active_segment is None:
            st.info("Selectionne un segment dans le graphe pour afficher ses actions d'edition.")
        else:
            active_segment_index = int(selected_segment_index)
            current_label = str(active_segment.get("label") or "autre")
            current_start_s = float(active_segment.get("start_offset_s", 0.0))
            current_end_s = float(active_segment.get("end_offset_s", 0.0))
            current_duration_s = float(active_segment.get("duration_s", current_end_s - current_start_s))
            signature = (
                active_segment_index,
                current_label,
                round(current_start_s, 3),
                round(current_end_s, 3),
            )
            selection_signature_key = f"selected_segment_signature_{session_id}"
            active_type_key = f"selected_segment_type_{session_id}"
            active_start_key = f"selected_segment_start_{session_id}"
            active_end_key = f"selected_segment_end_{session_id}"
            new_segment_label_key = f"new_temporal_segment_label_{session_id}"
            if st.session_state.get(selection_signature_key) != signature:
                st.session_state[selection_signature_key] = signature
                st.session_state[active_type_key] = current_label if current_label in TEMPORAL_SEGMENT_LABELS else "autre"
                st.session_state[active_start_key] = float(round(current_start_s, 1))
                st.session_state[active_end_key] = float(round(current_end_s, 1))
            if new_segment_label_key not in st.session_state:
                st.session_state[new_segment_label_key] = current_label if current_label in TEMPORAL_SEGMENT_LABELS else "technique"

            st.markdown(
                f"**Segment selectionne** : `{current_label}` de `{format_offset_label(current_start_s)}` a `{format_offset_label(current_end_s)}` ({format_duration(current_duration_s)})"
            )

            edit_cols = st.columns([1.15, 0.9, 0.9, 0.95])
            with edit_cols[0]:
                st.selectbox(
                    "Type du segment",
                    options=TEMPORAL_SEGMENT_LABELS,
                    key=active_type_key,
                )
            with edit_cols[1]:
                st.number_input(
                    "Debut (s)",
                    min_value=0.0,
                    max_value=float(total_duration_s),
                    step=1.0,
                    key=active_start_key,
                )
            with edit_cols[2]:
                st.number_input(
                    "Fin (s)",
                    min_value=0.0,
                    max_value=float(total_duration_s),
                    step=1.0,
                    key=active_end_key,
                )
            with edit_cols[3]:
                st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
                if st.button(
                    "Appliquer",
                    key=f"apply_fc_segment_changes_{session_id}_{active_segment_index}",
                    use_container_width=True,
                ):
                    if update_selected_temporal_segment(
                        session_id,
                        total_duration_s,
                        repository,
                        label=st.session_state.get(active_type_key, current_label),
                        start_offset_s=float(st.session_state.get(active_start_key, current_start_s)),
                        end_offset_s=float(st.session_state.get(active_end_key, current_end_s)),
                    ):
                        st.session_state["activity_management_notice"] = {
                            "level": "success",
                            "message": f"Segment mis a jour pour {session_id}.",
                        }
                        st.rerun()

            st.caption("Ajouts et suppression : utilise les boutons ci-dessous apres avoir selectionne un segment. La timeline se recale automatiquement pour rester continue.")
            insert_cols = st.columns([1.0, 1.0, 1.0, 1.0])
            with insert_cols[0]:
                st.selectbox(
                    "Type du nouveau segment",
                    options=TEMPORAL_SEGMENT_LABELS,
                    key=new_segment_label_key,
                )
            with insert_cols[1]:
                if st.button(
                    "Ajouter a gauche",
                    key=f"add_fc_segment_left_{session_id}_{active_segment_index}",
                    use_container_width=True,
                ):
                    if add_temporal_segment(
                        session_id,
                        st.session_state.get(new_segment_label_key, "technique"),
                        total_duration_s,
                        repository,
                        side="left",
                    ):
                        st.rerun()
            with insert_cols[2]:
                if st.button(
                    "Ajouter a droite",
                    key=f"add_fc_segment_right_{session_id}_{active_segment_index}",
                    use_container_width=True,
                ):
                    if add_temporal_segment(
                        session_id,
                        st.session_state.get(new_segment_label_key, "technique"),
                        total_duration_s,
                        repository,
                        side="right",
                    ):
                        st.rerun()
            with insert_cols[3]:
                if st.button(
                    "Supprimer",
                    key=f"delete_fc_segment_{session_id}_{active_segment_index}",
                    use_container_width=True,
                ):
                    if delete_temporal_segment(session_id, total_duration_s, repository):
                        st.rerun()

    with action_cols[1]:
        render_section_label("Validation")
        if st.button(
            "Valider segmentation",
            key=f"validate_fc_segments_{session_id}",
            use_container_width=True,
            disabled=not segments,
        ):
            if persist_temporal_segments(session_id, segments, total_duration_s, repository):
                segments = st.session_state.get(temporal_segments_key(session_id), [])
                st.session_state["activity_management_notice"] = {
                    "level": "success",
                    "message": f"Segmentation temporelle validee pour {session_id}.",
                }
                st.rerun()

        export_frame = repository.build_fc_phase_segments_export_frame(segments)
        export_csv = export_frame.to_csv(index=False) if not export_frame.empty else "segment_id,type,t_debut_s,t_fin_s,duree_s\n"
        st.download_button(
            "Exporter CSV",
            data=export_csv,
            file_name=f"{session_id}_segments_fc.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"export_fc_segments_{session_id}",
        )

    render_section_label("Resume des phases")
    summary_frame = repository.summarize_fc_phase_segments(st.session_state.get(temporal_segments_key(session_id), []))
    if summary_frame.empty:
        st.info("Aucun segment temporel a resumer.")
    else:
        summary_frame = summary_frame.copy()
        summary_frame["debut"] = summary_frame["debut_s"].apply(format_offset_label)
        summary_frame["fin"] = summary_frame["fin_s"].apply(format_offset_label)
        summary_frame["duree"] = summary_frame["duree_s"].apply(format_duration)
        st.dataframe(summary_frame[["ordre", "phase", "debut", "fin", "duree", "origine"]], use_container_width=True, hide_index=True)


def initialize_form_state(session, repository: ProcessedSessionRepository) -> None:
    selected_session_key = "activity_form_session_id"
    if st.session_state.get(selected_session_key) == session.session_id:
        return

    session_id = session.session_id
    st.session_state[selected_session_key] = session_id
    st.session_state[f"annotation_input_{session_id}"] = session.annotation or session.session_id
    st.session_state[f"activity_family_{session_id}"] = session.activity_family or ""
    st.session_state[f"activity_label_{session_id}"] = session.activity_label or ""
    st.session_state[f"activity_notes_{session_id}"] = session.activity_notes or ""
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

    phase_items: list[dict[str, Any]] = []
    for index, stored_phase in enumerate(session.judo_phases or [], start=1):
        phase_uid = stored_phase.get("phase_uid") or f"phase_{index}"
        phase_label = stored_phase.get("phase_label") or stored_phase.get("label") or "echauffement"
        phase_items.append({"phase_uid": phase_uid, "phase_label": phase_label})
    st.session_state[f"phase_items_{session_id}"] = phase_items
    st.session_state[f"phase_counter_{session_id}"] = len(phase_items)

    randori_blocks = {block.get("phase_index"): block for block in (session.judo_randori_blocks or [])}
    for phase_index, phase_item in enumerate(phase_items):
        phase_uid = phase_item["phase_uid"]
        phase_label = phase_item["phase_label"]
        st.session_state[f"phase_label_{session_id}_{phase_uid}"] = phase_label
        block = randori_blocks.get(phase_index, {})
        kind_default = str(block.get("randori_kind") or default_randori_kind(phase_label))
        count_default = str(block.get("randori_count") or "NA")
        rest_default = str(block.get("rest_between_randoris_min") or "NA")
        duration_default = str(block.get("randori_duration_min") or ("NA" if kind_default == "libres" else "4.0"))
        st.session_state[f"randori_kind_{session_id}_{phase_uid}"] = kind_default
        st.session_state[f"randori_count_{session_id}_{phase_uid}"] = count_default
        st.session_state[f"randori_rest_{session_id}_{phase_uid}"] = rest_default
        st.session_state[f"randori_duration_{session_id}_{phase_uid}"] = duration_default


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
    st.session_state[f"randori_duration_{session_id}_{phase_uid}"] = "NA" if phase_label == "randoris libres" else "4.0"


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


def build_phase_payload(session_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    phases: list[dict[str, Any]] = []
    randori_blocks: list[dict[str, Any]] = []
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
        randori_blocks.append(
            {
                "phase_index": phase_index,
                "phase_uid": phase_uid,
                "randori_kind": st.session_state.get(f"randori_kind_{session_id}_{phase_uid}", default_randori_kind(phase_label)),
                "randori_count": to_optional_number(st.session_state.get(f"randori_count_{session_id}_{phase_uid}"), integer=True),
                "randori_duration_min": to_optional_number(st.session_state.get(f"randori_duration_{session_id}_{phase_uid}"), integer=False),
                "rest_between_randoris_min": to_optional_number(st.session_state.get(f"randori_rest_{session_id}_{phase_uid}"), integer=False),
            }
        )
    return phases, randori_blocks


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

    cols = st.columns(5)
    cols[0].metric("Duree", format_duration(session.duree_s))
    cols[1].metric("FC min brute", f"{fc_min} bpm")
    cols[2].metric("FC max brute", f"{fc_max} bpm")
    cols[3].metric("Heure debut", format_datetime_label(start_dt))
    cols[4].metric("Heure fin", format_datetime_label(end_dt))


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
    render_section_label("Activite selectionnee")
    st.subheader(session.annotation or session.session_id)
    st.caption(f"Session ID : {session.session_id} | {session.date} | debut {session.heure_debut}")
    render_status_pills(session)
    render_session_summary(session, hr_frame, repository)

    render_section_label("Edition")
    session_id = session.session_id
    annotation_key = f"annotation_input_{session_id}"
    family_key = f"activity_family_{session_id}"
    label_key = f"activity_label_{session_id}"
    notes_key = f"activity_notes_{session_id}"
    judo_type_key = f"judo_session_type_{session_id}"

    form_col1, form_col2 = st.columns(2)
    with form_col1:
        st.text_input("Nom de l'activite", key=annotation_key)
        st.selectbox("Famille d'activite", options=["", "judo", "prepa", "autres"], key=family_key)

    selected_family = st.session_state.get(family_key)
    current_label = st.session_state.get(label_key, "")
    with form_col2:
        if selected_family == "judo":
            st.selectbox("Type de seance judo", options=["technique", "randoris"], key=judo_type_key)
            selected_judo_type = st.session_state.get(judo_type_key)
            if selected_judo_type == "technique":
                st.session_state[label_key] = "judo > technique"
                st.info("Activite attribuee automatiquement : judo > technique")
            else:
                randori_options = [
                    option for option in activity_options
                    if option.startswith("judo > randoris >")
                ]
                if current_label not in randori_options:
                    st.session_state[label_key] = randori_options[0] if randori_options else ""
                st.selectbox("Categorie precise", options=randori_options, key=label_key)
        elif selected_family == "prepa":
            st.session_state[judo_type_key] = "technique"
            prepa_options = [
                option for option in activity_options
                if option.startswith("prepa >")
            ]
            if current_label not in prepa_options:
                st.session_state[label_key] = prepa_options[0] if prepa_options else ""
            st.selectbox("Activite precise", options=prepa_options, key=label_key)
        elif selected_family == "autres":
            st.session_state[judo_type_key] = "technique"
            st.session_state[label_key] = "autres"
            st.info("Activite attribuee automatiquement : autres")
        else:
            st.session_state[judo_type_key] = "technique"
            st.session_state[label_key] = ""

    st.text_area("Notes generales", key=notes_key, height=100)

    if st.session_state.get(family_key) == "judo" and st.session_state.get(judo_type_key) == "randoris":
        render_section_label("Phases judo")
        phase_items = st.session_state.get(f"phase_items_{session_id}", [])
        add_col1, add_col2 = st.columns([3, 1])
        new_phase_key = f"new_phase_{session_id}"
        if new_phase_key not in st.session_state:
            st.session_state[new_phase_key] = judo_phase_options[0] if judo_phase_options else "echauffement"
        with add_col1:
            st.selectbox("Ajouter une phase", options=judo_phase_options or ["echauffement"], key=new_phase_key)
        with add_col2:
            if st.button("Ajouter", key=f"add_phase_button_{session_id}", use_container_width=True):
                append_phase(session_id, st.session_state.get(new_phase_key, "echauffement"))
                st.rerun()

        if not phase_items:
            st.info("Ajoute les phases de la seance pour decrire l'ordre : echauffement, technique, randoris TW, technique, etc.")

        for index, phase_item in enumerate(phase_items):
            phase_uid = phase_item["phase_uid"]
            phase_label_key = f"phase_label_{session_id}_{phase_uid}"
            row_cols = st.columns([0.8, 3.4, 0.8, 0.8, 0.9])
            row_cols[0].markdown(f"**{index + 1}.**")
            row_cols[1].selectbox(
                "Phase",
                options=judo_phase_options or ["echauffement"],
                key=phase_label_key,
                label_visibility="collapsed",
            )
            if row_cols[2].button("Monter", key=f"move_up_{session_id}_{phase_uid}", use_container_width=True):
                move_phase(session_id, phase_uid, -1)
                st.rerun()
            if row_cols[3].button("Descendre", key=f"move_down_{session_id}_{phase_uid}", use_container_width=True):
                move_phase(session_id, phase_uid, 1)
                st.rerun()
            if row_cols[4].button("Retirer", key=f"remove_phase_{session_id}_{phase_uid}", use_container_width=True):
                remove_phase(session_id, phase_uid)
                st.rerun()

            current_phase_label = st.session_state.get(phase_label_key, phase_item["phase_label"])
            if not phase_is_randori(current_phase_label):
                continue

            render_section_label(f"Details randoris phase {index + 1}")
            kind_key = f"randori_kind_{session_id}_{phase_uid}"
            count_key = f"randori_count_{session_id}_{phase_uid}"
            duration_key = f"randori_duration_{session_id}_{phase_uid}"
            rest_key = f"randori_rest_{session_id}_{phase_uid}"
            if kind_key not in st.session_state:
                st.session_state[kind_key] = default_randori_kind(current_phase_label)
            detail_cols = st.columns(4)
            detail_cols[0].selectbox("Categorie randori", options=RANDORI_KIND_OPTIONS, key=kind_key)
            detail_cols[1].selectbox("Nombre de randoris", options=RANDORI_COUNT_OPTIONS, key=count_key)
            duration_options = ["NA"] + RANDORI_DURATION_OPTIONS if st.session_state.get(kind_key) == "libres" else RANDORI_DURATION_OPTIONS
            if st.session_state.get(duration_key) not in duration_options:
                st.session_state[duration_key] = duration_options[0]
            detail_cols[2].selectbox("Duree d'un randori (min)", options=duration_options, key=duration_key)
            detail_cols[3].selectbox("Repos entre randoris (min)", options=RANDORI_REST_OPTIONS, key=rest_key)

    action_cols = st.columns(3)
    if action_cols[0].button("Enregistrer", type="primary", use_container_width=True):
        payload: dict[str, Any] = {
            "annotation": st.session_state.get(annotation_key),
            "activity_family": st.session_state.get(family_key) or None,
            "activity_label": st.session_state.get(label_key) or None,
            "activity_notes": st.session_state.get(notes_key) or None,
        }
        if st.session_state.get(family_key) == "judo":
            payload["judo_session_type"] = st.session_state.get(judo_type_key)
            if st.session_state.get(judo_type_key) == "randoris":
                judo_phases, judo_randori_blocks = build_phase_payload(session_id)
                payload["judo_phases"] = judo_phases
                payload["judo_randori_blocks"] = judo_randori_blocks
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

    action_cols[2].metric("Statut annotation", "Oui" if session.is_activity_annotated else "Non")
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


