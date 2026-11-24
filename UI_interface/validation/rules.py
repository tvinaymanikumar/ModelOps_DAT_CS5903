"""Post-extraction rule registry.

Rules NEVER modify values; they return flags. Any flag puts the item in
`needs_review`. Rule specs are strings: "name" or "name:arg1:arg2".
They are generated from the task (field types, required flags and the
review options in the UI) by rules_for_task(). Add a rule with the @rule
decorator; no UI change is needed.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable


Flag = dict[str, str]
RuleFunc = Callable[..., list[Flag]]
RULES: dict[str, RuleFunc] = {}


def rule(name: str) -> Callable[[RuleFunc], RuleFunc]:
    def deco(func: RuleFunc) -> RuleFunc:
        RULES[name] = func
        return func
    return deco


def _flag(rule_name: str, detail: str, field: str = "") -> Flag:
    return {"rule": rule_name, "field": field, "detail": detail}


def _is_empty(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


@rule("required_not_empty")
def required_not_empty(result: dict[str, Any], field: str) -> list[Flag]:
    if field in result and _is_empty(result.get(field)):
        return [_flag("required_not_empty", "Required field is empty", field)]
    return []


@rule("digits_only")
def digits_only(result: dict[str, Any], field: str) -> list[Flag]:
    v = result.get(field)
    if _is_empty(v):
        return []
    if not re.fullmatch(r"\d+", str(v).strip()):
        return [_flag("digits_only", f"'{v}' contains non-digit characters", field)]
    return []


@rule("year_range")
def year_range(result: dict[str, Any], field: str, lo: str = "1900", hi: str = "2100") -> list[Flag]:
    v = result.get(field)
    if _is_empty(v):
        return []
    s = str(v).strip()
    if not re.fullmatch(r"\d{4}", s):
        return [_flag("year_range", f"'{s}' is not a 4-digit year", field)]
    if not int(lo) <= int(s) <= int(hi):
        return [_flag("year_range", f"{s} is outside {lo}-{hi}", field)]
    return []


_DATE_FORMATS = ["%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y", "%d.%m.%Y", "%Y-%m-%d", "%m/%d/%y",
                 "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%d/%m/%Y %H:%M"]


@rule("date_parseable")
def date_parseable(result: dict[str, Any], field: str) -> list[Flag]:
    v = result.get(field)
    if _is_empty(v):
        return []
    s = str(v).strip()
    for fmt in _DATE_FORMATS:
        try:
            datetime.strptime(s, fmt)
            return []
        except ValueError:
            continue
    return [_flag("date_parseable", f"'{s}' is not a recognised date", field)]


@rule("numeric")
def numeric(result: dict[str, Any], field: str) -> list[Flag]:
    v = result.get(field)
    if _is_empty(v):
        return []
    cleaned = re.sub(r"(?i)(rm|rs\.?|\$|₹|,|\s)", "", str(v))
    try:
        float(cleaned)
        return []
    except ValueError:
        return [_flag("numeric", f"'{v}' is not a number", field)]


@rule("no_newlines")
def no_newlines(result: dict[str, Any], field: str) -> list[Flag]:
    v = result.get(field)
    if isinstance(v, str) and ("\n" in v or "\\n" in v):
        return [_flag("no_newlines", "Value contains line breaks; expected one line", field)]
    return []


@rule("non_empty_ratio")
def non_empty_ratio(result: dict[str, Any], max_empty_share: str = "0.5") -> list[Flag]:
    if not result:
        return [_flag("non_empty_ratio", "No fields extracted")]
    empty = sum(1 for v in result.values() if _is_empty(v))
    share = empty / len(result)
    if share > float(max_empty_share):
        return [_flag("non_empty_ratio", f"{empty} of {len(result)} fields are empty")]
    return []


@rule("schema_free_min_fields")
def schema_free_min_fields(result: dict[str, Any], minimum: str = "3") -> list[Flag]:
    if len(result) < int(minimum):
        return [_flag("schema_free_min_fields", f"Only {len(result)} fields extracted (expected at least {minimum})")]
    return []


@rule("transcription_not_empty")
def transcription_not_empty(result: dict[str, Any]) -> list[Flag]:
    text = str(result.get("transcription", ""))
    if not text.strip():
        return [_flag("transcription_not_empty", "Transcription is empty")]
    if text.count("[illegible]") >= 3:
        return [_flag("transcription_not_empty", f"{text.count('[illegible]')} words marked illegible")]
    return []


def run_rules(result: dict[str, Any], specs: list[str]) -> list[Flag]:
    flags: list[Flag] = []
    for spec in specs:
        name, *args = spec.split(":")
        func = RULES.get(name)
        if func is None:
            flags.append(_flag("unknown_rule", f"Rule '{name}' is not registered"))
            continue
        try:
            flags.extend(func(result, *args))
        except Exception as exc:  # a broken rule must never break processing
            flags.append(_flag(name, f"Rule error: {type(exc).__name__}: {exc}"))
    return flags


def rules_for_task(mode: str, fields: list[dict[str, Any]] | None = None, *, flag_required: bool = True,
                   check_types: bool = True, max_empty_share: float | None = 0.5,
                   min_fields: int | None = 3) -> list[str]:
    """Build rule specs from the user's task definition (dataset independent)."""
    specs: list[str] = []
    if mode == "transcription":
        return ["transcription_not_empty"]
    if mode == "schema_based":
        for f in fields or []:
            if ":" in f["key"]:
                continue  # rule args are ':'-separated; skip keys that contain ':'
            if flag_required and f.get("required", True):
                specs.append(f"required_not_empty:{f['key']}")
            if check_types and f.get("type") == "date":
                specs.append(f"date_parseable:{f['key']}")
    elif mode == "schema_free" and min_fields:
        specs.append(f"schema_free_min_fields:{int(min_fields)}")
    if max_empty_share is not None:
        specs.append(f"non_empty_ratio:{max_empty_share}")
    return specs
