"""Extraction helpers taken from the team's pipeline scripts and made
dataset independent.

Script origin of each function:
  parse_json_response, flatten_json, normalized_key   pipeline_batch_schema.py
  enforce_schema, add_schema_instructions,
  ensure_candidate_keys                               pipeline_batch_schema.py (SCHEMA -> user fields)
  normalize_integer  (was normalize_cheque_amount)    pipeline_batch_cheques.py, for any "integer" field
  normalize_age / normalize_sex /
  normalize_age_sex  (Nigeria medical records)        pipeline_batch_NMRDS.py, optional

In the scripts SCHEMA was a hard-coded dict per dataset. Here the schema is
the field list the user types in the UI, and each field's type decides how its
value is written: text fields as strings, "integer" fields as JSON integers
(the cheque `amount` rule), "number" fields as JSON numbers.
"""
from __future__ import annotations

import json
import re
from typing import Any

OCR_CHAR_LIMIT = 12000        # paddle_text[:12000] in the scripts
CANDIDATE_CHAR_LIMIT = 10000  # qwen_candidate_text[:10000]


# ---- JSON parsing ----------------------------------------------------------

def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a JSON object, allowing Markdown fences and surrounding text.
    Raises ValueError when no JSON object is present."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```")).strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        result = None
        for position, character in enumerate(text):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[position:])
                if isinstance(candidate, dict):
                    result = candidate
                    break
            except json.JSONDecodeError:
                continue
        if result is None:
            raise ValueError("No valid JSON object found in model response.")
    if not isinstance(result, dict):
        raise ValueError("Expected the model response to be a JSON object.")
    return result


def flatten_json(data: dict[str, Any], prefix: str = "", separator: str = ".") -> dict[str, str]:
    """Flatten nested JSON into dot-separated keys and string values."""
    flattened: dict[str, str] = {}
    for key, value in data.items():
        full_key = f"{prefix}{separator}{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_json(value, full_key, separator))
        elif isinstance(value, list):
            flattened[full_key] = json.dumps(value, ensure_ascii=False)
        elif value is None:
            flattened[full_key] = ""
        elif isinstance(value, bool):
            flattened[full_key] = str(value).lower()
        else:
            flattened[full_key] = str(value).strip()
    return flattened


def normalized_key(key: Any) -> str:
    """Normalize formatting differences in keys, not their meanings."""
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


# ---- value normalizers -----------------------------------------------------

def normalize_integer(value: Any) -> int | str:
    """'1,80,000\\-' or '1,80,000/-' -> 180000. Blank/unreadable -> ''.
    (normalize_cheque_amount in pipeline_batch_cheques.py)"""
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return value
    raw = str(value).strip()
    if not raw:
        return ""
    digits = re.sub(r"[^0-9]", "", re.sub(r"\.0+$", "", raw))
    return int(digits) if digits else ""


def normalize_number(value: Any) -> float | int | str:
    """'Rs. 1,200.50' -> 1200.5. Blank/unreadable -> ''."""
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return value
    raw = re.sub(r"[^0-9.\-]", "", str(value).strip())
    if not raw or raw in {".", "-"}:
        return ""
    try:
        number = float(raw)
    except ValueError:
        return ""
    return int(number) if number.is_integer() else number


def normalize_sex(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    token = re.sub(r"[^a-z]", "", raw.lower())
    if token in {"m", "male"}:
        return "male"
    if token in {"f", "female", "fem"}:
        return "female"
    return raw


def normalize_age(value: Any) -> str:
    """Numeric ages -> '<number> years' / '<number> months' (no unit = years)."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(.*?)\s*", raw, flags=re.IGNORECASE)
    if not m:
        return raw
    number, unit_text = m.group(1), re.sub(r"\bold\b", "", m.group(2).lower().strip())
    unit = re.sub(r"[\s./_-]+", "", unit_text).strip()
    if not unit:
        return f"{number} years"
    if unit in {"y", "yr", "yrs", "year", "years", "yo", "yro"}:
        return f"{number} years"
    if unit in {"m", "mo", "mos", "mon", "mons", "month", "months", "mth", "mths", "monthold", "monthsold"} \
            or "month" in unit or "mon" in unit:
        return f"{number} months"
    return raw


def normalize_age_sex(fields: dict[str, Any], age_key: str, sex_key: str) -> dict[str, Any]:
    """Normalize age and sex, including combined entries such as '12/f'."""
    out = dict(fields)
    age_value = str(out.get(age_key, "") or "").strip()
    sex_value = str(out.get(sex_key, "") or "").strip()
    combined = re.fullmatch(r"\s*(\d+(?:\.\d+)?\s*(?:(?:years?|yrs?|y|months?|mons?|mon|mos?|m))?)"
                            r"\s*[/|]\s*(m|male|f|female|fem)\s*", age_value, flags=re.IGNORECASE)
    if combined:
        age_value = combined.group(1).strip()
        if not sex_value:
            sex_value = combined.group(2)
    out[age_key] = normalize_age(age_value)
    out[sex_key] = normalize_sex(sex_value)
    return out


def age_sex_keys(template: dict[str, Any] | None) -> tuple[str, str] | None:
    """Return the actual (age, sex) keys if the user's fields contain both."""
    if not template:
        return None
    lookup = {normalized_key(k): k for k in template}
    if "age" in lookup and "sex" in lookup:
        return lookup["age"], lookup["sex"]
    return None


# ---- schema handling -------------------------------------------------------

def _field_types(fields: list[dict[str, Any]] | None) -> dict[str, str]:
    return {f["key"]: f.get("type", "string") for f in (fields or [])}


def enforce_schema(fields: dict[str, Any] | None, template: dict[str, Any] | None,
                   field_specs: list[dict[str, Any]] | None = None,
                   normalize_age_and_sex: bool = False) -> dict[str, Any]:
    """Keep exactly the schema keys (in schema order), fill missing keys with "",
    and write each value according to its field type. Key matching ignores case,
    spaces, hyphens and underscores; it does not guess synonyms."""
    if template is None:
        return dict(fields or {})
    fields = fields or {}
    types = _field_types(field_specs)
    lookup: dict[str, Any] = {}
    for key, value in fields.items():
        lookup.setdefault(normalized_key(key), value)
    output: dict[str, Any] = {}
    for key in template:
        value = fields[key] if key in fields else lookup.get(normalized_key(key), "")
        ftype = types.get(key, "string")
        if ftype == "integer":
            value = normalize_integer(value)
        elif ftype == "number":
            value = normalize_number(value)
        elif ftype == "boolean" and isinstance(value, bool):
            pass
        elif value is None:
            value = ""
        elif isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            value = str(value).lower()
        else:
            value = str(value).strip()
        output[key] = value
    if normalize_age_and_sex:
        keys = age_sex_keys(template)
        if keys:
            output = normalize_age_sex(output, *keys)
    return output


def add_schema_instructions(base_prompt: str, template: dict[str, Any] | None,
                            field_specs: list[dict[str, Any]] | None = None,
                            normalize_age_and_sex: bool = False) -> str:
    """Append the REQUIRED PREDEFINED SCHEMA block from the scripts.
    The value rule is generated from the field types (cheque rule generalised)."""
    if template is None:
        return base_prompt
    types = _field_types(field_specs)
    ints = [k for k in template if types.get(k) == "integer"]
    nums = [k for k in template if types.get(k) == "number"]
    if ints or nums:
        lines = []
        if ints:
            lines.append(f"The field(s) {', '.join(ints)} must be JSON integers, not strings. For example, "
                         'convert "1,80,000\\-" or "1,80,000/-" to 180000 by removing commas and trailing notation.')
        if nums:
            lines.append(f"The field(s) {', '.join(nums)} must be JSON numbers, not strings, without currency "
                         "symbols or thousands separators.")
        lines.append("All other fields must be JSON strings. If a numeric field is blank or unreadable, "
                     'return an empty string "".')
        value_rule = "\n".join(lines)
    else:
        value_rule = "Every value must be a string."
    hints = [f"- {f['key']}: {f['description']}" for f in (field_specs or []) if f.get("description")]
    block = (
        "\n\nREQUIRED PREDEFINED SCHEMA:\n"
        f"{json.dumps(template, ensure_ascii=False, indent=2)}\n\n"
        "Return exactly these schema keys, with exactly the same spelling.\n"
        "Do not add keys outside the schema.\n"
        f"{value_rule}\n"
        'If a field is blank, absent, or unreadable, return an empty string "".\n'
        "Do not guess or fabricate values. Return one valid JSON object only.\n"
    )
    if hints:
        block += "\nFIELD HINTS:\n" + "\n".join(hints) + "\n"
    if normalize_age_and_sex and age_sex_keys(template):
        block += (
            "\nAGE AND SEX NORMALIZATION:\n"
            '- If sex is M or m, return "male". If sex is F or f, return "female".\n'
            '  Also normalize the clear words "male", "female", or "fem" accordingly.\n'
            "- Format ages as '<number> years' or '<number> months'.\n"
            '- Examples: 5, 5 y, 5 yr, 5 year, 5 years -> "5 years".\n'
            '- Examples: 5 m, 5 mo, 5 mon, 5 month, 5 months -> "5 months".\n'
            "- If an age is numeric and has no unit, assume years.\n"
            '- If age and sex are combined, e.g. "12/f", split them into age "12 years"\n'
            '  and sex "female". Use the visible image as the source of truth.\n'
        )
    return base_prompt.rstrip() + block


def ensure_candidate_keys(final_fields: dict[str, Any], qwen_candidate: dict[str, Any] | None,
                          template: dict[str, Any] | None, field_specs: list[dict[str, Any]] | None = None,
                          normalize_age_and_sex: bool = False) -> dict[str, Any]:
    """Apply the schema, or (no schema) keep Qwen-only keys missing from the final answer as ""."""
    if template is not None:
        return enforce_schema(final_fields, template, field_specs, normalize_age_and_sex)
    completed = dict(final_fields or {})
    for key in (qwen_candidate or {}):
        completed.setdefault(key, "")
    return completed


def fill_pipeline_prompt(prompt: str, paddle_text: str, qwen_candidate: dict[str, Any] | None) -> str:
    candidate = json.dumps(qwen_candidate or {}, indent=2, ensure_ascii=False)
    return (prompt.replace("{paddle_text}", (paddle_text or "")[:OCR_CHAR_LIMIT])
                  .replace("{qwen_candidate}", candidate[:CANDIDATE_CHAR_LIMIT]))
