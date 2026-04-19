from __future__ import annotations

import os

import streamlit.components.v1 as components

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_FRONTEND_COMPONENT_PATH = os.path.join(_PROJECT_ROOT, "frontend", "rr_clean_viewer")

rr_clean_viewer = components.declare_component(
    "rr_clean_viewer",
    path=_FRONTEND_COMPONENT_PATH,
)
