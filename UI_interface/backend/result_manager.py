"""Read saved jobs back from disk (previous runs, evaluation comparisons)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from config.settings import OUTPUTS_DIR, UPLOADS_DIR


def list_jobs() -> list[dict[str, Any]]:
    jobs = []
    for d in sorted(OUTPUTS_DIR.iterdir(), reverse=True) if OUTPUTS_DIR.exists() else []:
        meta = d / "summary" / "job.json"
        if d.is_dir() and meta.exists():
            try:
                info = json.loads(meta.read_text(encoding="utf-8"))
            except ValueError:
                continue
            info["n_results"] = len(list((d / "json").glob("*.json")))
            info["has_evaluation"] = (d / "evaluation" / "metrics.json").exists()
            jobs.append(info)
    return jobs


def load_envelopes(job_id: str) -> list[dict[str, Any]]:
    folder = OUTPUTS_DIR / job_id / "json"
    out = []
    for f in sorted(folder.glob("*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except ValueError:
            continue
    return out


def load_metrics(job_id: str) -> dict[str, Any] | None:
    path = OUTPUTS_DIR / job_id / "evaluation" / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def delete_job(job_id: str) -> None:
    for root in (OUTPUTS_DIR, UPLOADS_DIR):
        target = (root / job_id).resolve()
        if target.parent == root.resolve() and target.exists():
            shutil.rmtree(target)
