"""Session-state helpers shared by frontend modules."""
from __future__ import annotations

import streamlit as st

from backend.processing_manager import ProcessingManager


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("manager", ProcessingManager())
    ss.setdefault("pages", [])            # [(name, page, PIL image)]
    ss.setdefault("thumbs", [])           # [(label, png bytes)]
    ss.setdefault("upload_errors", [])
    ss.setdefault("upload_signature", None)
    ss.setdefault("external_confirmed", False)
    ss.setdefault("was_running", False)
    ss.setdefault("results_job_id", None)


def manager() -> ProcessingManager:
    return st.session_state["manager"]


STATUS_COLORS = {
    "pending": "gray", "running": "blue", "completed": "green", "needs_review": "orange",
    "parse_failed": "red", "failed": "red", "cancelled": "gray",
}

STATUS_LABELS = {
    "pending": "Pending", "running": "Running", "completed": "Completed", "needs_review": "Needs review",
    "parse_failed": "Parse failed", "failed": "Failed", "cancelled": "Cancelled",
}
