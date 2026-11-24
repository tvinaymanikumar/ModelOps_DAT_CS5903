"""Key matching between ground-truth and predicted JSON:
normalized exact match first, then a one-to-one fuzzy lexical fallback."""
from __future__ import annotations

import re

from rapidfuzz import fuzz


def normalize_key(key: str) -> str:
    k = str(key).lower()
    k = re.sub(r"[^\w\s]", " ", k)
    k = k.replace("_", " ")
    return re.sub(r"\s+", " ", k).strip()


def match_keys(gt_keys: list[str], pred_keys: list[str], fuzzy_threshold: float = 0.85) -> dict[str, str]:
    """Return {gt_key: pred_key}. Each predicted key is used at most once."""
    mapping: dict[str, str] = {}
    pred_norm: dict[str, list[str]] = {}
    for pk in pred_keys:
        pred_norm.setdefault(normalize_key(pk), []).append(pk)
    used: set[str] = set()
    # 1) normalized exact
    for gk in gt_keys:
        for pk in pred_norm.get(normalize_key(gk), []):
            if pk not in used:
                mapping[gk] = pk
                used.add(pk)
                break
    # 2) fuzzy fallback, best pairs first
    remaining_gt = [g for g in gt_keys if g not in mapping]
    remaining_pred = [p for p in pred_keys if p not in used]
    candidates = []
    for gk in remaining_gt:
        for pk in remaining_pred:
            score = fuzz.ratio(normalize_key(gk), normalize_key(pk)) / 100.0
            if score >= fuzzy_threshold:
                candidates.append((score, gk, pk))
    for score, gk, pk in sorted(candidates, key=lambda c: -c[0]):
        if gk not in mapping and pk not in used:
            mapping[gk] = pk
            used.add(pk)
    return mapping
