"""Single entry point for JSON handling used by processors and the UI.

Implementation lives in validation/ (parser, schema validator, rules); this
module gives the backend one stable interface:

    parse(raw_text)                         -> (dict | None, error | None)
    generate_template(schema)               -> {"key": "", ...}
    validate_result(result, template, schema, strict_keys=True)
                                            -> ValidationReport
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from validation.json_parser import flatten, parse_json_object
from validation.schema_validator import coerce_types, complete_template, validate


@dataclass
class ValidationReport:
    result: dict[str, Any]
    schema_valid: bool
    errors: list[str] = field(default_factory=list)
    missing_keys_added: list[str] = field(default_factory=list)
    extra_keys: dict[str, Any] = field(default_factory=dict)
    coerced_fields: list[str] = field(default_factory=list)


def parse(raw_text: str, duplicate_strategy: str = "suffix") -> tuple[dict[str, Any] | None, str | None]:
    """Parse model text into a JSON object. Garbage returns (None, error), never {}."""
    obj, err, _ = parse_json_object(raw_text, duplicate_strategy)
    return obj, err


def generate_template(schema: dict[str, Any]) -> dict[str, Any]:
    """Simple JSON template ({"key": ""}) from a formal JSON Schema."""
    return {key: "" for key in schema.get("properties", {})}


def validate_result(result: dict[str, Any], template: dict[str, Any], schema: dict[str, Any] | None,
                    strict_keys: bool = True, missing_value: Any = "") -> ValidationReport:
    flat = flatten(result) if any(isinstance(v, dict) for v in result.values()) else result
    completed, missing, extra = complete_template(flat, template, strict_keys, missing_value)
    coerced, changed = coerce_types(completed, schema)
    errors = validate(coerced, schema) if strict_keys else []
    return ValidationReport(result=coerced, schema_valid=not errors, errors=errors, missing_keys_added=missing,
                            extra_keys={k: flat[k] for k in extra}, coerced_fields=changed)
