"""Document Extraction: upload images, choose schema or no schema, extract.

Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

try:
    from backend.router import RoutingError, resolve_processor_path
    from config.settings import APP_TITLE
    from frontend.evaluation_ui import render_evaluation
    from frontend.processing_ui import render_controls, render_progress
    from frontend.prompt_ui import render_prompt_editor
    from frontend.results_ui import render_history, render_job_results
    from frontend.schema_ui import render_schema_choice
    from frontend.settings_ui import render_settings
    from frontend.sidebar import render_sidebar
    from frontend.state import init_state, manager
    from frontend.upload_ui import render_upload
except ImportError as exc:  # incomplete or mixed installation: explain instead of a traceback
    st.set_page_config(page_title="Setup problem", page_icon="⚠️")
    st.error(f"The app could not start: {exc}")
    st.markdown(
        "This usually means some project files are missing, often after copying a new version over an "
        "older folder, or that the requirements are not installed.\n\n"
        "1. Extract the ZIP into a **new, empty folder**.\n"
        "2. Double-click `setup_windows.bat` in that folder.\n"
        "3. Run `python check_setup.py`; it lists exactly what is missing."
    )
    st.stop()

st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; max-width: 1200px; }
  h1 { font-weight: 650; letter-spacing: -0.01em; }
  h3 { margin-top: 0.4rem; }
  [data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

init_state()
settings = render_sidebar()

st.title("Document extraction")
tab_extract, tab_history, tab_eval, tab_settings = st.tabs(["Extract", "Earlier runs", "Evaluation", "Settings"])

with tab_extract:
    st.subheader("1. Add images")
    render_upload(settings)

    st.subheader("2. Schema")
    task, blockers = render_schema_choice(settings["transcribe"])
    prompt = render_prompt_editor(task)

    st.subheader("3. Extract")
    if not settings["model_name"]:
        blockers.append("no model supports this task")
    elif not settings["model_ok"]:
        blockers.append(f"set up the model ({settings['model_reason']})")
    else:
        try:
            resolve_processor_path(task.mode, settings["model_name"], settings["processor_choice"])
        except RoutingError as exc:
            blockers.append(str(exc))
    if not settings["external_ok"]:
        blockers.append("confirm in the sidebar that documents may be sent to the external provider")
    if not prompt.strip():
        blockers.append("the prompt is empty")
    st.caption(f"Model: {settings['model_name'] or 'none'} (change it in the sidebar)")
    render_controls(settings, prompt, task, blockers)
    render_progress()

    job = manager().job
    if job is not None and not manager().is_running:
        st.subheader("Results")
        render_job_results(job.job_id)

with tab_history:
    render_history()

with tab_eval:
    render_evaluation()

with tab_settings:
    render_settings()
