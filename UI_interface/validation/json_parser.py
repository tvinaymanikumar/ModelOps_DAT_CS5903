"""Robust JSON extraction from raw model text.

Handles markdown fences, leading/trailing prose, trailing commas, single top-level
lists of key/value pairs, duplicate keys (renamed, not dropped) and nested
objects (optionally flattened with a separator).
"""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


class DuplicateKeyRenamer:
    """object_pairs_hook that keeps every duplicate key.

    strategy "suffix": NAME, NAME_2, NAME_3 ... (reading order)
    strategy "list":   NAME -> ["first", "second"]
    """

    def __init__(self, strategy: str = "suffix"):
        self.strategy = strategy
        self.renamed: list[str] = []

    def __call__(self, pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        counts: dict[str, int] = {}
        for key, value in pairs:
            if key not in out:
                out[key] = value
                counts[key] = 1
                continue
            if self.strategy == "list":
                if not isinstance(out[key], list) or counts[key] == 1:
                    out[key] = [out[key]]
                out[key].append(value)
                counts[key] += 1
            else:
                counts[key] += 1
                new_key = f"{key}_{counts[key]}"
                while new_key in out:
                    counts[key] += 1
                    new_key = f"{key}_{counts[key]}"
                out[new_key] = value
            self.renamed.append(key)
        return out


def _candidates(text: str) -> list[str]:
    cands: list[str] = []
    for m in _FENCE.finditer(text):
        cands.append(m.group(1).strip())
    cands.append(text.strip())
    # first balanced {...} or [...] block, string-aware
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        cands.append(text[start:i + 1])
                        break
            else:
                break
            start = text.find(opener, start + 1) if not cands else -1
    return cands


def _pairs_to_dict(value: list[Any]) -> dict[str, Any] | None:
    """[{"key": k, "value": v}, ...] or [[k, v], ...] -> dict."""
    out: dict[str, Any] = {}
    for item in value:
        if isinstance(item, dict) and {"key", "value"} <= set(item):
            k, v = str(item["key"]), item["value"]
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            k, v = str(item[0]), item[1]
        elif isinstance(item, dict) and len(item) == 1:
            k, v = next(iter(item.items()))
        else:
            return None
        base, n = k, 2
        while k in out:
            k = f"{base}_{n}"
            n += 1
        out[k] = v
    return out


def parse_json_object(text: str, duplicate_strategy: str = "suffix") -> tuple[dict[str, Any] | None, str | None, list[str]]:
    """Return (object, error, renamed_duplicate_keys). Never returns {} for garbage."""
    if not text or not text.strip():
        return None, "Empty model response", []
    last_error = "No JSON object found in model response"
    seen: set[str] = set()
    for cand in _candidates(text):
        if cand in seen or not cand:
            continue
        seen.add(cand)
        for attempt in (cand, _TRAILING_COMMA.sub(r"\1", cand)):
            hook = DuplicateKeyRenamer(duplicate_strategy)
            try:
                value = json.loads(attempt, object_pairs_hook=hook)
            except json.JSONDecodeError as exc:
                last_error = f"Invalid JSON: {exc.msg} (line {exc.lineno}, col {exc.colno})"
                continue
            if isinstance(value, dict):
                return value, None, hook.renamed
            if isinstance(value, list):
                as_dict = _pairs_to_dict(value)
                if as_dict is not None:
                    return as_dict, None, hook.renamed
                if len(value) == 1 and isinstance(value[0], dict):
                    return value[0], None, hook.renamed
                last_error = "Model returned a JSON list, not an object"
            else:
                last_error = f"Model returned a JSON {type(value).__name__}, not an object"
    return None, last_error, []


def flatten(obj: dict[str, Any], separator: str = " - ", prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts/lists: {"A": {"B": "x"}} -> {"A - B": "x"}."""
    out: dict[str, Any] = {}
    for key, value in obj.items():
        full = f"{prefix}{separator}{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten(value, separator, full))
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            for i, v in enumerate(value, 1):
                out.update(flatten(v, separator, f"{full}{separator}{i}"))
        elif isinstance(value, list):
            out[full] = ", ".join("" if v is None else str(v) for v in value)
        else:
            out[full] = "" if value is None else value
    return out
