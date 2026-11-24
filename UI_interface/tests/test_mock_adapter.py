import threading

from PIL import Image

from models.mock_adapter import MockAdapter
from processors.base_processor import ProcessingContext, render_prompt
from processors.schema_based import SchemaBasedProcessor
from processors.schema_free import SchemaFreeProcessor
from processors.transcription import TranscriptionProcessor
from tests.helpers import make_task, prompt_for

IMG = Image.new("RGB", (64, 64), "white")


def _ctx(task, adapter):
    return ProcessingContext(task=task, adapter=adapter, model_name="Mock Model", prompt=prompt_for(task.mode),
                             params={}, cancel_event=threading.Event())


def test_schema_based_returns_exactly_user_fields():
    fields = [{"key": "Vehicle No", "type": "string", "required": True, "description": ""},
              {"key": "odometer", "type": "integer", "required": True, "description": ""}]
    task = make_task("schema_based", fields)
    out = SchemaBasedProcessor().process(IMG, _ctx(task, MockAdapter(delay_s=0)))
    assert list(out.result) == ["Vehicle No", "odometer"]
    assert out.status in ("completed", "needs_review")


def test_prompt_contains_user_fields():
    task = make_task("schema_based")
    text = render_prompt(prompt_for("schema_based"), task.template, task.fields)
    assert '"policy_number": ""' in text and "REQUIRED PREDEFINED SCHEMA" in text
    assert "age, claim_amount" not in text and "integers" in text and "JSON numbers" in text


def test_schema_free_dedup_and_flatten():
    out = SchemaFreeProcessor().process(IMG, _ctx(make_task("schema_free"), MockAdapter(delay_s=0)))
    assert "NAME_2" in out.result and any(k.startswith("Educational Qualifications.1. Matric") for k in out.result)


def test_transcription():
    out = TranscriptionProcessor().process(IMG, _ctx(make_task("transcription"), MockAdapter(delay_s=0)))
    assert out.result["transcription"] and out.raw_text


def test_invalid_json_is_parse_failed():
    out = SchemaBasedProcessor().process(IMG, _ctx(make_task("schema_based"), MockAdapter(delay_s=0, invalid_json_rate=1.0)))
    assert out.status == "parse_failed" and out.result is None and out.raw_text


def test_empty_required_field_needs_review():
    out = SchemaBasedProcessor().process(IMG, _ctx(make_task("schema_based"), MockAdapter(delay_s=0, empty_rate=1.0)))
    assert out.status == "needs_review"
    assert any(f["rule"] == "required_not_empty" for f in out.flags)
