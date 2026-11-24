"""Load ground-truth uploads (individual files or a ZIP) keyed by file stem."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import PurePosixPath
from typing import Any


def _stem(name: str) -> str:
    return PurePosixPath(name.replace("\\", "/")).stem


def load_ground_truth(files: list[tuple[str, bytes]], fmt: str) -> tuple[dict[str, Any], list[str]]:
    """Return ({stem: gt}, errors). fmt is 'json' or 'txt'."""
    entries: list[tuple[str, bytes]] = []
    for name, data in files:
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist():
                    if not info.is_dir() and not info.filename.startswith("__MACOSX"):
                        entries.append((info.filename, zf.read(info)))
        else:
            entries.append((name, data))
    out: dict[str, Any] = {}
    errors: list[str] = []
    for name, data in entries:
        ext = name.lower().rsplit(".", 1)[-1]
        if ext != fmt:
            continue
        text = data.decode("utf-8-sig", errors="replace")
        if fmt == "json":
            try:
                value = json.loads(text)
            except ValueError as exc:
                errors.append(f"{name}: invalid JSON ({exc})")
                continue
            if not isinstance(value, dict):
                errors.append(f"{name}: ground truth must be a JSON object")
                continue
            out[_stem(name)] = value
        else:
            out[_stem(name)] = text
    return out, errors


def match_stem(output_stem: str, source_name: str, gt: dict[str, Any]) -> str | None:
    """Ground truth is matched by output stem (e.g. form_p2) or the source file stem."""
    for key in (output_stem, _stem(source_name)):
        if key in gt:
            return key
    lower = {k.lower(): k for k in gt}
    for key in (output_stem.lower(), _stem(source_name).lower()):
        if key in lower:
            return lower[key]
    return None
