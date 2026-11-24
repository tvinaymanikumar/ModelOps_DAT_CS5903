"""Sidebar: model and output format up front; everything else under Advanced."""
from __future__ import annotations

from typing import Any

import streamlit as st

from config.model_registry import availability, is_external, models_for_mode
from config.processor_registry import processors_for_mode
from config.settings import ALLOW_EXTERNAL_MODELS, DEFAULT_PDF_DPI, DEFAULT_RPM, REQUEST_TIMEOUT_S

OUTPUT_CHOICES = {"JSON + TXT + PDF": ["json", "txt", "pdf"], "JSON": ["json"], "TXT": ["txt"], "PDF": ["pdf"]}
GENERIC = "Generic (default)"


@st.cache_data(ttl=30, show_spinner=False)
def cached_availability(model_name: str) -> tuple[bool, str]:
    return availability(model_name)


def current_mode() -> str:
    """Mode from the page choices (read from session state so the model list matches)."""
    ss = st.session_state
    if ss.get("transcribe"):
        return "transcription"
    return "schema_based" if ss.get("schema_choice") == "With schema" else "schema_free"


def render_sidebar() -> dict[str, Any]:
    sb = st.sidebar
    mode = current_mode()
    models = models_for_mode(mode)
    status = {m: cached_availability(m) for m in models}

    def label(m: str) -> str:
        return f"{m}{' · external' if is_external(m) else ''}{'' if status[m][0] else ' (not set up)'}"

    if st.session_state.get("model_name") not in models:
        st.session_state["model_name"] = models[0] if models else None
    model_name = sb.selectbox("Model", models, format_func=label, key="model_name",
                              help="Start with Mock Model: it needs no GPU or API key.")
    ok, reason = status.get(model_name, (False, "No model supports this task"))
    if not ok:
        sb.warning(reason)

    external_ok = True
    if model_name and is_external(model_name):
        sb.error("This model is an external API. Images will leave your organization.")
        external_ok = sb.checkbox("I confirm these documents may be sent to an external provider",
                                  key="external_confirmed")

    out_label = sb.selectbox("Save results as", list(OUTPUT_CHOICES))
    output_formats = sorted(set(OUTPUT_CHOICES[out_label]) | {"json"})

    params: dict[str, Any] = {}
    adapter_options: dict[str, Any] = {}
    processor_choice = None
    with sb.expander("Advanced settings"):
        st.toggle("Handwriting transcription instead of extraction", key="transcribe",
                  help="Returns the handwritten text as plain text.")
        custom = processors_for_mode(mode)
        if custom:
            pick = st.selectbox("Processing pipeline", [GENERIC] + custom)
            processor_choice = None if pick == GENERIC else pick

        st.markdown("**Model**")
        params["temperature"] = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
        params["top_p"] = st.slider("Top-p", 0.0, 1.0, 1.0, 0.05)
        params["max_tokens"] = st.number_input("Max output tokens", 256, 16384, 3000, 256)
        params["max_image_side"] = st.number_input("Max image side sent to the model (px)", 256, 8000, 1600, 64,
                                                   help="Images are shrunk to this size and sent as JPEG.")
        params["guided_json"] = st.toggle("Constrained JSON decoding", value=False,
                                          disabled=mode != "schema_based",
                                          help="Sends the schema to models that support it (e.g. vLLM guided_json).")
        timeout_s = st.number_input("Request timeout (s)", 10, 1800, int(REQUEST_TIMEOUT_S), 10)
        max_retries = st.number_input("Retries on timeout / 429 / 5xx", 0, 10, 2)

        st.markdown("**Rate limits**")
        rpm = st.number_input("Requests per minute (0 = unlimited)", 0, 10000,
                              int(DEFAULT_RPM.get(model_name or "", 0)), key=f"rpm_{model_name}",
                              help="Retries count toward this limit.")
        min_delay = st.number_input("Minimum delay between requests (s)", 0.0, 120.0, 0.0, 0.5)

        # st.markdown("**Images**")
        # preprocess = {
        #     "deskew": st.toggle("Deskew small rotations", value=False),
        #     "contrast": st.toggle("Enhance contrast", value=False),
        #     "max_side": st.number_input("Max long side in pixels (0 = keep)", 0, 10000, 0, 256),
        # }
        # pdf_dpi = st.number_input("PDF render DPI", 72, 400, DEFAULT_PDF_DPI, 10)
        # gif_all = st.toggle("Process every GIF frame", value=False)
        preprocess = {"deskew": False, "contrast": False, "max_side": 0}
        pdf_dpi = DEFAULT_PDF_DPI
        gif_all = False

        st.markdown("**Post-processing**")
        params["normalize_age_sex"] = st.toggle(
            "Normalize age and sex fields", value=True,
            help="Only when your fields include both 'age' and 'sex': '12/f' -> age '12 years', sex 'female'.")
        if model_name == "Mock Model":
            st.markdown("**Mock model**")
            adapter_options = {
                "delay_s": st.slider("Seconds per image", 0.0, 5.0, 0.8, 0.1),
                "failure_rate": st.slider("Transient failure rate", 0.0, 1.0, 0.0, 0.05),
                "invalid_json_rate": st.slider("Invalid JSON rate", 0.0, 1.0, 0.0, 0.05),
                "empty_rate": st.slider("Empty field rate", 0.0, 1.0, 0.15, 0.05),
            }
        else:
            adapter_options = {"timeout_s": float(timeout_s)}
        if not ALLOW_EXTERNAL_MODELS:
            st.caption("External APIs are hidden (ALLOW_EXTERNAL_MODELS=false in .env).")

    params["preprocess"] = preprocess
    return {"mode": mode, "model_name": model_name, "model_ok": ok, "model_reason": reason,
            "processor_choice": processor_choice, "external_ok": external_ok, "output_formats": output_formats,
            "params": params, "rpm": rpm, "min_delay": min_delay, "max_retries": int(max_retries),
            "adapter_options": adapter_options, "pdf_dpi": int(pdf_dpi), "gif_all": gif_all,
            "transcribe": bool(st.session_state.get("transcribe"))}
