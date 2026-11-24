"""Metric primitives, as defined in the project report.

- Value exact match after normalization.
- ANLS: 1 - edit_distance / max(len) with the standard 0.5 threshold.
- CER / WER for transcription.
"""
from __future__ import annotations

import re
from typing import Any

from rapidfuzz.distance import Levenshtein


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    s = str(value).lower().replace("\\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" .,;:")
    return s


def value_exact(gt: Any, pred: Any) -> bool:
    return normalize_value(gt) == normalize_value(pred)


def nls(gt: Any, pred: Any, threshold: float = 0.5) -> float:
    a, b = normalize_value(gt), normalize_value(pred)
    if not a and not b:
        return 1.0
    score = 1.0 - Levenshtein.distance(a, b) / max(len(a), len(b))
    return score if score >= threshold else 0.0


def cer(gt: str, pred: str) -> float:
    gt, pred = gt or "", pred or ""
    if not gt:
        return 0.0 if not pred else 1.0
    return Levenshtein.distance(gt, pred) / len(gt)


def wer(gt: str, pred: str) -> float:
    g, p = (gt or "").split(), (pred or "").split()
    if not g:
        return 0.0 if not p else 1.0
    return Levenshtein.distance(g, p) / len(g)


def safe_div(a: float, b: float) -> float | None:
    return a / b if b else None
