from __future__ import annotations

import os

import streamlit.components.v1 as components

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_FRONTEND_COMPONENT_PATH = os.path.join(_PROJECT_ROOT, "frontend", "fc_segment_editor")

fc_segment_editor = components.declare_component(
    "fc_segment_editor",
    path=_FRONTEND_COMPONENT_PATH,
)
