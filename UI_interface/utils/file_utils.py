from __future__ import annotations

import re
from pathlib import Path


def safe_stem(name: str, max_len: int = 80) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "file"
    return stem[:max_len]


def extension(name: str) -> str:
    return Path(name).suffix.lower().lstrip(".")
