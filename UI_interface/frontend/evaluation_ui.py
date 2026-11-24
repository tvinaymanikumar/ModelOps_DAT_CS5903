"""Evaluation tab: score a job against uploaded ground truth.

Extraction jobs (with or without schema) use the team's key-value metrics
(evaluation/kv_metrics.py, ported from Metric_new.py):

    Combined KV Precision / Recall / F1 (micro-averaged), whole-form NED
    similarity (macro-averaged) and Strict Form Accuracy.

There are no scoring settings: key alignment follows the job's mode
("schema" for With schema, "schemafree" for No schema) with the script's
default lexical key threshold of 0.80 and its default key aliases.
Transcription jobs keep CER / WER.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from backend.result_manager import list_jobs, load_envelopes, load_metrics
from config.settings import OUTPUTS_DIR
from config.task_config import TaskConfig
from evaluation import kv_metrics
from evaluation.evaluator import aggregate, evaluate_transcription
from evaluation.ground_truth import load_ground_truth, match_stem
from frontend.results_ui import job_picker

METRICS_VERSION = 2
KV_KEYS = ["combined_kv_precision_pct", "combined_kv_recall_pct", "combined_kv_f1_pct",
           "normalized_edit_distance_similarity_pct", "strict_form_accuracy_pct"]
NICE = {
    "combined_kv_precision_pct": "KV Precision",
    "combined_kv_recall_pct": "KV Recall",
    "combined_kv_f1_pct": "KV F1",
    "normalized_edit_distance_similarity_pct": "NED similarity",
    "strict_form_accuracy_pct": "Strict form accuracy",
    "cer": "CER", "wer": "WER", "line_exact_match": "Line exact match",
    "anls": "ANLS", "document_exact_match": "Document exact match",
}
TX_KEYS = ["cer", "wer", "line_exact_match", "anls", "document_exact_match"]


def _align_mode(job_mode: str) -> str:
    return "schema" if job_mode == "schema_based" else "schemafree"


def render_evaluation() -> None:
    job = job_picker("eval_job_picker", "Job to evaluate")
    if job is None:
        st.info("Process some documents first, then upload their ground truth here.")
        return
    ds = TaskConfig.from_dict(job["task"])
    fmt = "txt" if ds.mode == "transcription" else "json"
    st.caption(f"Upload ground truth as .{fmt} files named like the documents (for example cheque_017.{fmt}), "
               "or one ZIP of them. Multi-page files use the page suffix, e.g. form_p2.")
    files = st.file_uploader("Ground truth", type=[fmt, "zip"], accept_multiple_files=True, key=f"gt_{job['job_id']}")

    if st.button("Score this job", type="primary", disabled=not files):
        gt, errors = load_ground_truth([(f.name, f.getvalue()) for f in files], fmt)
        for e in errors:
            st.error(e)
        _score(job, ds, gt)

    metrics = load_metrics(job["job_id"])
    if metrics and metrics.get("metrics_version") == METRICS_VERSION:
        _show(job, ds, metrics)
    elif metrics:
        st.info("This job was scored with the old metrics. Upload the ground truth and score it again.")


def _score(job: dict, ds: TaskConfig, gt: dict) -> None:
    envelopes = load_envelopes(job["job_id"])
    rows, fields, unmatched, tx_docs = [], {}, [], []
    align = _align_mode(ds.mode)
    for env in envelopes:
        key = match_stem(env["output_stem"], env["image"], gt)
        if key is None:
            unmatched.append(env["output_stem"])
            continue
        if ds.mode == "transcription":
            pred = (env.get("result") or {}).get("transcription", "")
            doc = evaluate_transcription(env["output_stem"], str(gt[key]), pred)
            doc.status = env["status"]
            tx_docs.append(doc)
            continue
        pred = env.get("result")
        row = kv_metrics.evaluate_pair(gt[key], pred, mode=align)
        row = {"document": env["output_stem"], "status": env["status"],
               "prediction": "ok" if pred else "missing_prediction", **row}
        rows.append(row)
        fields[env["output_stem"]] = kv_metrics.field_rows(gt[key], pred, mode=align)

    used = {match_stem(e["output_stem"], e["image"], gt) for e in envelopes}
    unused_gt = [k for k in gt if k not in used]
    if not rows and not tx_docs:
        st.error("No ground-truth file matched a processed document. Check the file names.")
        return

    out = {"metrics_version": METRICS_VERSION, "job_id": job["job_id"], "task_name": job.get("task_name"),
           "mode": job["mode"], "model": job["model"],
           "settings": {"key_alignment_mode": align, "lexical_key_threshold": kv_metrics.KEY_THRESHOLD},
           "unmatched_predictions": unmatched, "unused_ground_truth": unused_gt}
    if ds.mode == "transcription":
        out["aggregate"] = aggregate(tx_docs, "transcription")
        out["documents"] = [d.summary() for d in tx_docs]
        out["fields"] = {}
    else:
        out["aggregate"] = kv_metrics.aggregate_rows(rows)
        out["documents"] = rows
        out["fields"] = fields

    folder = OUTPUTS_DIR / job["job_id"] / "evaluation"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(out["documents"]).to_csv(folder / "per_document_metrics.csv", index=False)
    pd.DataFrame([out["aggregate"]]).to_csv(folder / "dataset_summary.csv", index=False)
    if out["fields"]:
        pd.DataFrame([{"document": d, **f} for d, fs in out["fields"].items() for f in fs]).to_csv(
            folder / "fields.csv", index=False)
    st.success(f"Scored {len(rows) or len(tx_docs)} document(s).")


def _show(job: dict, ds: TaskConfig, metrics: dict) -> None:
    agg = metrics["aggregate"]
    if ds.mode == "transcription":
        cols = st.columns(len(TX_KEYS))
        for c, k in zip(cols, TX_KEYS):
            v = agg.get(k)
            c.metric(NICE[k], "not computed" if v is None else f"{v * 100:.2f}")
    else:
        st.markdown("**Dataset metrics (%)**")
        cols = st.columns(len(KV_KEYS))
        for c, k in zip(cols, KV_KEYS):
            c.metric(NICE[k], f"{agg.get(k, 0.0):.2f}")
        st.caption(f"{agg['documents_evaluated']} documents · TP {agg['combined_kv_tp']} · FP {agg['combined_kv_fp']} · "
                   f"FN {agg['combined_kv_fn']} · strict-correct documents {agg['strict_form_correct_documents']}. "
                   "Precision / recall / F1 are micro-averaged; NED is macro-averaged.")
    if metrics.get("unmatched_predictions"):
        st.warning(f"No ground truth for: {', '.join(metrics['unmatched_predictions'][:20])}")
    if metrics.get("unused_ground_truth"):
        st.warning(f"Ground truth without a prediction: {', '.join(metrics['unused_ground_truth'][:20])}")

    st.markdown("**Per document**")
    docs = pd.DataFrame(metrics["documents"])
    if ds.mode != "transcription" and not docs.empty:
        docs = docs.drop(columns=["strict_form_correct"], errors="ignore")
        docs = docs.rename(columns={k: NICE[k] for k in KV_KEYS})
    st.dataframe(docs, hide_index=True, height=min(360, 38 + 35 * len(docs)))

    if metrics.get("fields"):
        doc = st.selectbox("Field-level diff", list(metrics["fields"]), key=f"diff_{job['job_id']}")
        fdf = pd.DataFrame(metrics["fields"][doc])
        if not fdf.empty:
            colors = {"correct": "#DDEFE3", "wrong": "#F8DEDB", "missing": "#FBEBCF", "extra": "#E4E8F0"}
            st.dataframe(fdf.style.apply(lambda r: [f"background-color: {colors.get(r['outcome'], '')}"] * len(r),
                                         axis=1), hide_index=True)

    st.markdown("**Comparison with other scored jobs of the same task**")
    keys = ["cer", "wer"] if ds.mode == "transcription" else KV_KEYS
    rows = []
    for other in list_jobs():
        if other.get("mode") != job.get("mode") or other.get("task_name") != job.get("task_name"):
            continue
        m = load_metrics(other["job_id"])
        if not m or m.get("metrics_version") != METRICS_VERSION:
            continue
        a = m["aggregate"]
        row = {"job": other["job_id"], "model": other["model"]}
        for k in keys:
            v = a.get(k)
            row[NICE[k]] = None if v is None else (round(v * 100, 2) if ds.mode == "transcription" else v)
        rows.append(row)
    if len(rows) > 1:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Values in %. Jobs are compared when they share the task name and mode.")
    else:
        st.caption("Run the same task with another model and score it to compare models here.")
    st.download_button("Download metrics (JSON)", json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
                       f"{job['job_id']}_metrics.json", "application/json")
