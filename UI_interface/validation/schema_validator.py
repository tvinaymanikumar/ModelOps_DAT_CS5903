"""Field list <-> template <-> JSON Schema, plus template completion,
safe type coercion and validation.

A *template* is the simple `{"key": ""}` object shown to the model.
A *schema* is the formal JSON Schema used for validation and, where the
model supports it, constrained decoding. Both are generated from the field
list the user enters in the UI. Every typed field also accepts "" so an
illegible value can be left empty instead of guessed.
"""
from __future__ import annotations

import re
from typing import Any

from jsonschema import Draft202012Validator

FIELD_TYPES = ["string", "integer", "number", "date", "boolean"]
_EMPTY = {"type": "string", "maxLength": 0}
_KEY_OK = re.compile(r"^[^\n\r\t]{1,120}$")


# ---- field rows ----------------------------------------------------------

def clean_fields(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize editor rows; return (fields, problems)."""
    fields, problems, seen = [], [], set()
    for r in rows:
        key = str(r.get("key") or "").strip()
        if not key:
            continue
        if not _KEY_OK.match(key):
            problems.append(f"Field name '{key[:30]}' is too long or has line breaks")
            continue
        if key in seen:
            problems.append(f"Field '{key}' is listed twice")
            continue
        seen.add(key)
        ftype = r.get("type") if r.get("type") in FIELD_TYPES else "string"
        req = r.get("required")
        fields.append({"key": key, "type": ftype, "required": True if req is None or req == "" else bool(req),
                       "description": str(r.get("description") or "").strip()})
    return fields, problems


def fields_from_names(text: str) -> list[dict[str, Any]]:
    """'name, age, policy number' (comma or newline separated) -> string fields."""
    names = [n.strip() for n in re.split(r"[,\n;]", text or "") if n.strip()]
    return [{"key": n, "type": "string", "required": True, "description": ""} for n in names]


def _type_of_prop(prop: dict[str, Any]) -> str:
    types: set[str] = set()
    fmt = prop.get("format")
    stack = [prop]
    while stack:
        p = stack.pop()
        t = p.get("type")
        if isinstance(t, str):
            types.add(t)
        elif isinstance(t, list):
            types.update(t)
        fmt = fmt or p.get("format")
        stack += p.get("anyOf", []) + p.get("oneOf", [])
    if fmt == "date" or prop.get("x-field-type") == "date":
        return "date"
    for t in ("boolean", "integer", "number"):
        if t in types:
            return t
    return "string"


def fields_from_schema(schema: dict[str, Any]) -> list[dict[str, Any]]:
    props = schema.get("properties", {})
    required = set(schema.get("required", list(props)))
    return [{"key": k, "type": _type_of_prop(p), "required": k in required,
             "description": p.get("description", "")} for k, p in props.items()]


def fields_from_template(template: dict[str, Any]) -> list[dict[str, Any]]:
    """Infer types from sample values in a template: {"age": 0, "name": ""}."""
    rows = []
    for k, v in template.items():
        t = "boolean" if isinstance(v, bool) else "integer" if isinstance(v, int) else \
            "number" if isinstance(v, float) else "string"
        rows.append({"key": str(k), "type": t, "required": True, "description": ""})
    return rows


def fields_from_json(data: Any) -> list[dict[str, Any]]:
    """Accept a preset ({"fields": [...]}), a JSON Schema, or a plain template."""
    if isinstance(data, dict) and isinstance(data.get("fields"), list):
        return clean_fields(data["fields"])[0]
    if isinstance(data, dict) and isinstance(data.get("properties"), dict):
        return fields_from_schema(data)
    if isinstance(data, dict):
        return fields_from_template(data)
    if isinstance(data, list):
        return clean_fields([{"key": x} if isinstance(x, str) else x for x in data])[0]
    raise ValueError("Expected a field list, a JSON Schema or a JSON template")


def template_from_fields(fields: list[dict[str, Any]]) -> dict[str, Any]:
    return {f["key"]: "" for f in fields}


def _prop_for(f: dict[str, Any]) -> dict[str, Any]:
    t = f["type"]
    if t == "integer":
        prop: dict[str, Any] = {"anyOf": [{"type": "integer"}, _EMPTY]}
    elif t == "number":
        prop = {"anyOf": [{"type": "number"}, _EMPTY]}
    elif t == "boolean":
        prop = {"anyOf": [{"type": "boolean"}, _EMPTY]}
    elif t == "date":
        prop = {"type": "string", "x-field-type": "date"}
    else:
        prop = {"type": "string"}
    if f.get("description"):
        prop["description"] = f["description"]
    return prop


def schema_from_fields(fields: list[dict[str, Any]], title: str = "Extraction") -> dict[str, Any]:
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": title, "type": "object",
            "properties": {f["key"]: _prop_for(f) for f in fields},
            "required": [f["key"] for f in fields],   # every key must be present (value may be "")
            "additionalProperties": False}


def fields_prompt_block(fields: list[dict[str, Any]]) -> str:
    lines = []
    for f in fields:
        hint = {"integer": "whole number, digits only", "number": "number without currency symbols",
                "date": "date as written", "boolean": "true or false"}.get(f["type"], "text")
        desc = f" — {f['description']}" if f.get("description") else ""
        lines.append(f"- {f['key']} ({hint}){desc}")
    return "\n".join(lines)


# ---- result handling -----------------------------------------------------

def complete_template(result: dict[str, Any], template: dict[str, Any], strict_keys: bool = True,
                      missing_value: Any = "") -> tuple[dict[str, Any], list[str], list[str]]:
    """Ensure every template key exists, in template order.

    Returns (completed, missing_keys_added, extra_keys). With strict_keys,
    extra keys are removed (the caller keeps them in metadata).
    """
    lookup = {str(k).strip(): v for k, v in result.items()}
    lower = {k.lower(): k for k in lookup}
    loose = {re.sub(r"[\s_\-]+", "", k.lower()): k for k in lookup}
    completed: dict[str, Any] = {}
    missing: list[str] = []
    used: set[str] = set()
    for key in template:
        src = key if key in lookup else lower.get(key.lower()) or loose.get(re.sub(r"[\s_\-]+", "", key.lower()))
        if src is not None and src not in used:
            completed[key] = lookup[src]
            used.add(src)
        else:
            completed[key] = missing_value
            missing.append(key)
    extra = [k for k in lookup if k not in used]
    if not strict_keys:
        for k in extra:
            completed[k] = lookup[k]
    for k, v in list(completed.items()):
        if v is None:
            completed[k] = missing_value
    return completed, missing, extra


_NUM_CLEAN = re.compile(r"(?i)(rs\.?|inr|rm|usd|eur|₹|\$|€|£|/-|,|\s)")


def coerce_types(result: dict[str, Any], schema: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """Safe coercions only ("2,223,000" -> 2223000, "yes" -> True, 5.9 -> "5.9"
    for text fields). Unparseable values are left as-is so validation reports them."""
    if not schema:
        return result, []
    props = schema.get("properties", {})
    out, changed = dict(result), []
    for key, value in result.items():
        if key not in props:
            continue
        t = _type_of_prop(props[key])
        if isinstance(value, str) and not value.strip():
            continue
        new = value
        if t == "integer":
            if isinstance(value, str):
                c = re.sub(r"\.0+$", "", _NUM_CLEAN.sub("", value))
                if re.fullmatch(r"-?\d+", c):
                    new = int(c)
            elif isinstance(value, float) and value.is_integer():
                new = int(value)
        elif t == "number" and isinstance(value, str):
            try:
                new = float(_NUM_CLEAN.sub("", value))
            except ValueError:
                pass
        elif t == "boolean" and isinstance(value, str):
            low = value.strip().lower()
            if low in {"true", "yes", "y", "checked", "x", "✓"}:
                new = True
            elif low in {"false", "no", "n", "unchecked"}:
                new = False
        elif t in ("string", "date") and isinstance(value, (int, float)) and not isinstance(value, bool):
            new = str(value)
        if new is not value:
            out[key] = new
            changed.append(key)
    return out, changed


def validate(result: dict[str, Any], schema: dict[str, Any] | None) -> list[str]:
    if not schema:
        return []
    validator = Draft202012Validator(schema)
    errors = []
    for err in sorted(validator.iter_errors(result), key=lambda e: list(e.path)):
        where = "/".join(str(p) for p in err.path) or "(root)"
        msg = err.message
        if err.validator == "anyOf":
            msg = f"{err.instance!r} does not match the field type"
        errors.append(f"{where}: {msg}")
    return errors
