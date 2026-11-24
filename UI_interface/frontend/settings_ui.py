"""Settings tab: endpoint checks and privacy policy.

Saved jobs are not listed here; open them in the "Earlier runs" tab."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config.model_registry import MODEL_REGISTRY, availability, is_external
from config.settings import ALLOW_EXTERNAL_MODELS


def render_settings() -> None:
    st.markdown("**Model endpoints**")
    # st.caption(f"Endpoints and keys are read from {BASE_DIR / '.env'}. Restart the app after editing it.")
    st.caption("Endpoints and keys are read from the `.env` file in the project folder. ")
    if st.button("Test connections"):
        st.cache_data.clear()
        rows = []
        for name, spec in MODEL_REGISTRY.items():
            ok, reason = availability(name)
            rows.append({"Model": name, "Modes": ", ".join(spec.modes),
                         "Where it runs": "External API" if is_external(name) else "Self-hosted / local",
                         "Available": "Yes" if ok else "No", "Detail": reason})
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.markdown("**Privacy**")
    st.write("External models are " + ("allowed, with a per-session confirmation." if ALLOW_EXTERNAL_MODELS
                                       else "hidden. Set ALLOW_EXTERNAL_MODELS=true in .env to enable them."))
    st.caption("Logs contain job IDs, file names, model, status and latency only. Each job's audit log "
               "(summary/audit.jsonl) stores hashes, never document content.")
