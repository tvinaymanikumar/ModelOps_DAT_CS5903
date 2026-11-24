"""Results: table, per-item viewer (fields, JSON, raw, report) and downloads."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from backend.output_manager import build_zip, human_readable
from backend.result_manager import delete_job, list_jobs, load_envelopes
from config.settings import OUTPUTS_DIR, UPLOADS_DIR
from frontend.state import STATUS_COLORS, STATUS_LABELS


def job_picker(key: str, label: str = "Job") -> dict | None:
    jobs = list_jobs()
    if not jobs:
        return None
    ids = [j["job_id"] for j in jobs]
    default = st.session_state.get("results_job_id")
    index = ids.index(default) if default in ids else 0
    by_id = {j["job_id"]: j for j in jobs}
    chosen = st.selectbox(label, ids, index=index, key=key,
                          format_func=lambda i: f"{i} · {by_id[i].get('task_name', '')} · {by_id[i].get('mode', '')} · {by_id[i]['model']} · {by_id[i]['n_results']} result(s)")
    return by_id[chosen]


def render_history() -> None:
    job = job_picker("results_job_picker", "Earlier run")
    if job is None:
        st.info("No runs yet. Results of every run are kept here.")
        return
    render_job_results(job["job_id"], key_prefix="hist")
    with st.expander("Delete this run"):
        from frontend.state import manager
        running = manager().is_running and manager().job and manager().job.job_id == job["job_id"]
        if st.button("Delete its uploads and outputs", key=f"del_{job['job_id']}", disabled=bool(running)):
            delete_job(job["job_id"])
            st.toast(f"Deleted {job['job_id']}")
            st.rerun()


def render_job_results(job_id: str, key_prefix: str = "cur") -> None:
    envelopes = load_envelopes(job_id)
    if not envelopes:
        st.caption("Results appear here as each image finishes.")
        return

    table = pd.DataFrame([{
        "Document": e["image"] + (f" (page {e['page']})" if e.get("output_stem", "").endswith(f"_p{e['page']}") else ""),
        "Status": STATUS_LABELS.get(e["status"], e["status"]),
        "Flags": len((e.get("validation") or {}).get("flags") or []),
        "Fields": len(e["result"]) if isinstance(e.get("result"), dict) and e.get("mode") != "transcription" else None,
        "Latency (ms)": (e.get("metrics") or {}).get("latency_ms"),
        "Error": e.get("error") or "",
    } for e in envelopes])
    st.dataframe(table, hide_index=True, height=min(360, 38 + 35 * len(table)))

    zip_col, _ = st.columns([1, 3])
    zip_col.download_button("Download all results (ZIP)", data=build_zip(OUTPUTS_DIR / job_id),
                            file_name=f"{job_id}.zip", mime="application/zip", width="stretch",
                            key=f"{key_prefix}_zip_{job_id}")

    names = {e["output_stem"]: e for e in envelopes}
    stem = st.selectbox("Document", list(names), format_func=lambda s: f"{names[s]['image']} [{s}]",
                        key=f"{key_prefix}_doc_{job_id}")
    env = names[stem]
    _render_item(job_id, env, key_prefix)


def _render_item(job_id: str, env: dict, key_prefix: str = "cur") -> None:
    status = env["status"]
    st.badge(STATUS_LABELS.get(status, status), color=STATUS_COLORS.get(status, "gray"))
    if env.get("error"):
        st.error(env["error"])
    flags = (env.get("validation") or {}).get("flags") or []
    for f in flags:
        field = f" · {f['field']}" if f.get("field") else ""
        st.warning(f"{f['rule']}{field}: {f['detail']}")

    left, right = st.columns([2, 3])
    image_path = UPLOADS_DIR / job_id / f"{env['output_stem']}.png"
    with left:
        if image_path.exists():
            st.image(str(image_path), width="stretch")
        else:
            st.caption("Source image was deleted.")
    with right:
        tabs = st.tabs(["Readable", "JSON", "Raw response", "Details"])
        result = env.get("result")
        with tabs[0]:
            if env.get("mode") == "transcription" and result:
                st.text(result.get("transcription", ""))
            elif result:
                st.dataframe(pd.DataFrame({"Field": list(result), "Value": [str(v) for v in result.values()]}),
                             hide_index=True)
            else:
                st.caption("No parsed result. See the raw response.")
        with tabs[1]:
            st.json(result if result is not None else {}, expanded=True)
        with tabs[2]:
            raw_path = OUTPUTS_DIR / job_id / "raw" / f"{env['output_stem']}.txt"
            st.code(raw_path.read_text(encoding="utf-8") if raw_path.exists() else "", language=None)
        with tabs[3]:
            meta = dict(env.get("metadata") or {})
            dis = meta.pop("disagreements", None)
            if dis:
                st.markdown("**VLM vs OCR disagreements**")
                st.dataframe(pd.DataFrame(dis), hide_index=True)
            ocr_text = meta.pop("ocr_text", None)
            if ocr_text:
                with st.expander("OCR lines (text, confidence)"):
                    st.code(ocr_text, language=None)
            st.json({"model": env["model"], "processor": env.get("processor"), "metrics": env.get("metrics"),
                     "schema_errors": (env.get("validation") or {}).get("schema_errors"), "metadata": meta},
                    expanded=False)

    stem = env["output_stem"]
    root = OUTPUTS_DIR / job_id
    d1, d2, d3 = st.columns(3)
    d1.download_button("JSON", json.dumps(env, ensure_ascii=False, indent=2), f"{stem}.json",
                       "application/json", width="stretch", key=f"{key_prefix}_dl_json_{job_id}_{stem}")
    txt_path = root / "txt" / f"{stem}.txt"
    d2.download_button("TXT", txt_path.read_text(encoding="utf-8") if txt_path.exists() else human_readable(env),
                       f"{stem}.txt", "text/plain", width="stretch", key=f"{key_prefix}_dl_txt_{job_id}_{stem}")
    pdf_path = root / "pdf" / f"{stem}.pdf"
    if pdf_path.exists():
        d3.download_button("PDF", pdf_path.read_bytes(), f"{stem}.pdf", "application/pdf",
                           width="stretch", key=f"{key_prefix}_dl_pdf_{job_id}_{stem}")
    else:
        d3.button("PDF not generated", disabled=True, width="stretch", key=f"{key_prefix}_no_pdf_{job_id}_{stem}")
