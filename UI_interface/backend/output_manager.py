"""Writes per-item outputs immediately, plus job summary, audit log and ZIP.

outputs/<job_id>/
    json/<stem>.json      result envelope (always written: needed for results, retry, evaluation)
    raw/<stem>.txt        untouched model response (always written)
    txt/<stem>.txt        human-readable report or transcription   (if TXT selected)
    pdf/<stem>.pdf        ReportLab report                         (if PDF selected)
    summary/job.json, summary/summary.csv, summary/audit.jsonl
    evaluation/           written by the evaluation tab
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from backend.job import Item, Job
from config.settings import env
from processors.base_processor import ProcessOutput
from utils.pdf_report import build_item_pdf

SUBDIRS = ["json", "raw", "txt", "pdf", "summary", "evaluation"]


def human_readable(envelope: dict[str, Any]) -> str:
    lines = [f"Document: {envelope.get('image')}",
             f"Task:     {envelope.get('task_name')} ({envelope.get('mode')})",
             f"Model:    {envelope.get('model')}",
             f"Status:   {envelope.get('status')}", ""]
    result = envelope.get("result")
    if envelope.get("mode") == "transcription" and result:
        lines += ["Transcription", "-" * 40, result.get("transcription", "")]
    elif result:
        width = min(max((len(str(k)) for k in result), default=10), 60)
        lines += ["Extracted fields", "-" * 40]
        for k, v in result.items():
            lines.append(f"{str(k):<{width}} : {v}")
    else:
        lines.append(f"No result. {envelope.get('error') or ''}")
    flags = (envelope.get("validation") or {}).get("flags") or []
    if flags:
        lines += ["", "Review flags", "-" * 40]
        lines += [f"[{f.get('rule')}] {f.get('field') + ': ' if f.get('field') else ''}{f.get('detail')}" for f in flags]
    return "\n".join(lines) + "\n"


class OutputManager:
    def __init__(self, job: Job):
        self.job = job
        self.root = job.output_dir
        for d in SUBDIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)

    # ---- per item --------------------------------------------------------
    def save_item(self, item: Item, output: ProcessOutput) -> dict[str, Any]:
        mode = self.job.task.mode
        image_bytes = Path(item.image_path).read_bytes() if Path(item.image_path).exists() else b""
        envelope = {
            "job_id": self.job.job_id,
            "item_id": item.item_id,
            "image": item.source_name,
            "page": item.page,
            "output_stem": item.output_stem,
            "task_name": self.job.task.name,
            "mode": mode,
            "model": self.job.model_name,
            "processor": self.job.params.get("processor", ""),
            "status": output.status,
            "result": output.result,
            "error": output.error,
            "validation": {"schema_valid": not output.schema_errors if output.result is not None else False,
                           "schema_errors": output.schema_errors, "flags": output.flags},
            "metrics": {"latency_ms": round(output.latency_ms, 1) if output.latency_ms is not None else None,
                        "peak_memory_mb": output.peak_memory_mb, "attempts": item.attempts},
            "metadata": output.metadata,
            "prompt_sha256": hashlib.sha256((output.rendered_prompt or self.job.prompt).encode()).hexdigest(),
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        stem = item.output_stem
        (self.root / "json" / f"{stem}.json").write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.root / "raw" / f"{stem}.txt").write_text(output.raw_text or "", encoding="utf-8")
        if "txt" in self.job.output_formats:
            text = output.result.get("transcription", "") + "\n" if (mode == "transcription" and output.result) else human_readable(envelope)
            (self.root / "txt" / f"{stem}.txt").write_text(text, encoding="utf-8")
        if "pdf" in self.job.output_formats:
            try:
                build_item_pdf(self.root / "pdf" / f"{stem}.pdf", envelope, item.image_path)
            except Exception as exc:  # PDF failure must not lose the JSON result
                envelope.setdefault("metadata", {})["pdf_error"] = f"{type(exc).__name__}: {exc}"
        self._audit(envelope)
        return envelope

    def _audit(self, env_: dict[str, Any]) -> None:
        host = ""
        for prefix, key in (("qwen", "QWEN_BASE_URL"), ("nuextract", "NUEXTRACT_BASE_URL"), ("glm", "GLM_BASE_URL")):
            if env(key) and env_["model"].lower().startswith(prefix):
                host = urlparse(env(key)).hostname or ""
        record = {"ts": env_["created_at"], "job_id": env_["job_id"], "item_id": env_["item_id"],
                  "model": env_["model"], "endpoint_host": host, "external": env_["model"] == "Gemini",
                  "prompt_sha256": env_["prompt_sha256"], "image_sha256": env_["image_sha256"],
                  "status": env_["status"], "latency_ms": env_["metrics"]["latency_ms"]}
        with (self.root / "summary" / "audit.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # ---- per job ---------------------------------------------------------
    def write_job_settings(self) -> None:
        (self.root / "summary" / "job.json").write_text(
            json.dumps(self.job.settings_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def write_summary(self) -> Path:
        path = self.root / "summary" / "summary.csv"
        with self.job.lock:
            rows = [(it.item_id, it.source_name, it.page, it.output_stem, it.status.value, len(it.flags),
                     "" if it.latency_ms is None else round(it.latency_ms, 1), it.attempts, it.error or "")
                    for it in self.job.items]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "file", "page", "output_stem", "status", "flags", "latency_ms", "attempts", "error"])
            w.writerows(rows)
        self.write_job_settings()
        return path


def build_zip(job_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for sub in SUBDIRS:
            folder = job_dir / sub
            if not folder.exists():
                continue
            for f in sorted(folder.rglob("*")):
                if f.is_file():
                    zf.write(f, f"{job_dir.name}/{sub}/{f.relative_to(folder).as_posix()}")
    return buf.getvalue()
