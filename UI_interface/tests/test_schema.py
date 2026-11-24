import json

import pytest

from config.settings import PRESETS_DIR
from tests.helpers import CLAIM_FIELDS
from validation.json_parser import flatten, parse_json_object
from validation.schema_validator import (clean_fields, coerce_types, complete_template, fields_from_json,
                                         fields_from_names, fields_prompt_block, schema_from_fields,
                                         template_from_fields, validate)


def test_fields_from_names():
    rows = fields_from_names("name, age\npolicy_number; claim_amount")
    assert [r["key"] for r in rows] == ["name", "age", "policy_number", "claim_amount"]


def test_clean_fields_flags_duplicates_and_blanks():
    fields, problems = clean_fields([{"key": "a"}, {"key": " "}, {"key": "a"}, {"key": "b", "type": "weird"}])
    assert [f["key"] for f in fields] == ["a", "b"] and fields[1]["type"] == "string"
    assert problems and "twice" in problems[0]


def test_template_and_schema_from_fields():
    t = template_from_fields(CLAIM_FIELDS)
    assert t == {"name": "", "age": "", "policy_number": "", "claim_amount": "", "date_of_incident": ""}
    s = schema_from_fields(CLAIM_FIELDS)
    assert s["required"] == list(t) and s["additionalProperties"] is False


def test_roundtrip_schema_template_and_preset():
    s = schema_from_fields(CLAIM_FIELDS)
    back = fields_from_json(s)
    assert [(f["key"], f["type"]) for f in back] == [(f["key"], f["type"]) for f in CLAIM_FIELDS]
    assert [f["type"] for f in fields_from_json({"a": "", "n": 0, "x": 1.5, "b": False})] == \
        ["string", "integer", "number", "boolean"]
    for preset in PRESETS_DIR.glob("*.json"):
        assert fields_from_json(json.loads(preset.read_text()))


def test_validation_and_coercion():
    s = schema_from_fields(CLAIM_FIELDS)
    r = {"name": "A", "age": "47", "policy_number": "P1", "claim_amount": "Rs. 1,200.50", "date_of_incident": ""}
    r2, changed = coerce_types(r, s)
    assert r2["age"] == 47 and r2["claim_amount"] == 1200.5 and validate(r2, s) == []
    r2["age"] = "forty"
    assert validate(r2, s)
    r2["age"] = ""  # empty is allowed for every type
    assert validate(r2, s) == []


def test_complete_template_strict_and_loose_key_match():
    t = template_from_fields(CLAIM_FIELDS)
    out, missing, extra = complete_template({"Name": "A", "policy number": "P1", "doctor": "x"}, t, True)
    assert out["name"] == "A" and out["policy_number"] == "P1"
    assert "age" in missing and extra == ["doctor"] and "doctor" not in out


def test_prompt_block_mentions_types():
    block = fields_prompt_block(CLAIM_FIELDS)
    assert "age (whole number" in block and "Claimant name" in block


def test_parser_cases():
    assert parse_json_object('Here:\n```json\n{"a": "1",}\n```')[0] == {"a": "1"}
    assert parse_json_object("I could not read it.")[0] is None
    assert parse_json_object('{"N": 1, "N": 2}')[0] == {"N": 1, "N_2": 2}
    assert flatten({"A": {"B": {"C": "x"}}}) == {"A - B - C": "x"}


def test_backend_json_validator_facade():
    from backend.json_validator import generate_template, parse, validate_result
    s = schema_from_fields(CLAIM_FIELDS)
    assert generate_template(s) == template_from_fields(CLAIM_FIELDS)
    obj, err = parse('{"name": "A", "age": 30, "x": 1}')
    rep = validate_result(obj, template_from_fields(CLAIM_FIELDS), s)
    assert rep.schema_valid and "x" in rep.extra_keys and "policy_number" in rep.missing_keys_added
