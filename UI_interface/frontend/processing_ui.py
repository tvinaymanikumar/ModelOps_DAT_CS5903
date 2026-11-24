"""Start / Stop / Clear / Retry controls and live progress."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from backend.job import Job
from backend.processing_manager import add_items
from backend.router import RoutingError
from config.task_config import TaskConfig
from frontend.state import STATUS_LABELS, manager


def _start_new_job(settings: dict[str, Any], prompt: str, task: TaskConfig) -> None:
    mgr = manager()
    job = Job(task=task, model_name=settings["model_name"], prompt=prompt, params=dict(settings["params"]),
              output_formats=settings["output_formats"], processor_choice=settings["processor_choice"])
    add_items(job, st.session_state["pages"])
    mgr.set_job(job)
    _launch(settings)
    st.session_state["results_job_id"] = job.job_id


def _launch(settings: dict[str, Any]) -> None:
    mgr = manager()
    try:
        started = mgr.start(rpm=settings["rpm"], min_delay_s=settings["min_delay"], max_retries=settings["max_retries"],
                            adapter_options=settings["adapter_options"])
    except (RoutingError, PermissionError, KeyError) as exc:
        st.error(str(exc))
        return
    if started:
        st.session_state["was_running"] = True


def render_controls(settings: dict[str, Any], prompt: str, task: TaskConfig | None, blockers: list[str]) -> None:
    mgr = manager()
    running = mgr.is_running
    job = mgr.job
    n = len(st.session_state["pages"])
    retryable = job is not None and any(it.status.value in ("failed", "parse_failed", "cancelled") for it in job.items)

    c1, c2, c3 = st.columns([2, 1, 1])
    label = f"Extract from {n} image{'s' if n != 1 else ''}" if n else "Extract"
    start = c1.button(label, type="primary", width="stretch", disabled=running or not n or bool(blockers))
    stop = c2.button("Stop", width="stretch", disabled=not running or mgr.stop_requested,
                     help="Finish the current image, then stop.")
    retry = c3.button("Retry failed", width="stretch", disabled=running or not retryable)
    if not n:
        st.caption("Add images in step 1 to start.")
    for b in blockers:
        st.caption(f"To start: {b}.")

    if start:
        _start_new_job(settings, prompt, task)
        st.rerun()
    if stop:
        mgr.stop()
        st.toast("Stopping after the current image")
        st.rerun()
    if retry:
        if mgr.reset_retryable():
            _launch(settings)
        st.rerun()


def render_progress() -> None:
    mgr = manager()

    @st.fragment(run_every=1.0 if mgr.is_running else None)
    def _live() -> None:
        job = mgr.job
        if job is None:
            return
        counts = job.counts()
        total = len(job.items)
        done = total - counts["pending"] - counts["running"]
        state = "Stopping after the current image" if mgr.stop_requested else \
            "Processing" if mgr.is_running else "Finished"
        st.progress(done / total if total else 0.0, text=f"{state}: {done} of {total} image(s)")

        m = st.columns(5)
        m[0].metric("Done", counts["completed"])
        m[1].metric("Needs review", counts["needs_review"])
        m[2].metric("Failed", counts["failed"] + counts["parse_failed"])
        m[3].metric("Remaining", counts["pending"] + counts["running"])
        m[4].metric("Cancelled", counts["cancelled"])
        if not mgr.is_running:
            if st.session_state.get("was_running"):  # just finished: redraw the page to show results
                st.session_state["was_running"] = False
                st.rerun(scope="app")
            return  # finished: the results below replace the live table

        left, right = st.columns([1, 3])
        current = job.get_item(job.current_item_id) if job.current_item_id else None
        with left:
            if current is not None:
                st.image(current.image_path, caption=f"Now: {current.display_name}", width="stretch")
            else:
                st.caption(f"Job {job.job_id}")
        with right:
            rows = job.snapshot()
            df = pd.DataFrame(rows)
            if not df.empty:
                df["status"] = df["status"].map(STATUS_LABELS)
                df["latency_ms"] = df["latency_ms"].map(lambda v: None if v is None or pd.isna(v) else round(v))
                st.dataframe(df[["name", "status", "flags", "latency_ms", "attempts", "error"]], hide_index=True,
                             height=min(420, 38 + 35 * len(df)),
                             column_config={"name": "Document", "status": "Status", "flags": "Flags",
                                            "latency_ms": "Latency (ms)", "attempts": "Attempts", "error": "Error"})
            lat = [r["latency_ms"] for r in rows if r["latency_ms"]]
            if lat:
                st.caption(f"Mean latency {sum(lat) / len(lat):,.0f} ms over {len(lat)} image(s)")

        if st.session_state.get("was_running") and not mgr.is_running:
            st.session_state["was_running"] = False
            st.rerun(scope="app")

    _live()
