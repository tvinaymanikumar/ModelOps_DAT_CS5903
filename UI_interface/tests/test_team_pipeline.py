"""Tests for the processors built from the team's scripts, using a scripted
adapter (no network) and a local fake OpenAI-compatible server for QwenAdapter."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

from models.base_adapter import BaseAdapter, ModelResponse
from models.dat_endpoints import image_to_base64
from processors.base_processor import ProcessingContext, render_prompt
from processors.custom.qwen_scripts import QwenOnlyProcessor
from processors.dat_extraction import (add_schema_instructions, enforce_schema, flatten_json, normalize_age_sex,
                                       normalize_integer, parse_json_response)
from tests.helpers import make_task, prompt_for

IMG = Image.new("RGB", (40, 30), "white")
CHEQUE = [{"key": "payee_name", "type": "string", "required": True, "description": ""},
          {"key": "amount", "type": "integer", "required": True, "description": ""},
          {"key": "year", "type": "string", "required": False, "description": ""}]
MEDICAL = [{"key": k, "type": "string", "required": True, "description": ""} for k in ("name", "age", "sex")]


class ScriptedAdapter(BaseAdapter):
    """Returns queued responses in order and records the prompts it received."""
    name = "scripted"

    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)
        self.prompts = []
        self.lock = threading.Lock()

    def check_available(self):
        return True, "ok"

    def generate(self, request, cancel_event=None):
        with self.lock:
            self.prompts.append(request.prompt)
            return ModelResponse(raw_text=self.responses.pop(0), latency_ms=5.0)


def ctx_for(task, adapter, **params):
    return ProcessingContext(task=task, adapter=adapter, model_name="t", prompt=prompt_for(task.mode),
                             params=params, cancel_event=threading.Event())


# ---- helpers from the scripts ------------------------------------------------

def test_parse_json_response_handles_fences_and_prose():
    assert parse_json_response('```json\n{"a": "1"}\n```') == {"a": "1"}
    assert parse_json_response('Result: {"a": {"b": 2}} done') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        parse_json_response("no json here")


def test_flatten_json_like_scripts():
    assert flatten_json({"a": {"b": 1}, "c": None, "d": True, "e": [1, 2]}) == \
        {"a.b": "1", "c": "", "d": "true", "e": "[1, 2]"}


def test_enforce_schema_cheque_integer_and_loose_keys():
    t = {"payee_name": "", "amount": "", "year": ""}
    out = enforce_schema({"Payee Name": " Kumar ", "AMOUNT": "1,80,000/-", "extra": "x"}, t, CHEQUE)
    assert out == {"payee_name": "Kumar", "amount": 180000, "year": ""}
    assert normalize_integer("1,80,000\\-") == 180000 and normalize_integer("") == ""


def test_age_sex_normalization_from_nmrds():
    t = {"name": "", "age": "", "sex": ""}
    out = enforce_schema({"name": "J.I", "age": "12/f", "sex": ""}, t, MEDICAL, normalize_age_and_sex=True)
    assert out == {"name": "J.I", "age": "12 years", "sex": "female"}
    assert normalize_age_sex({"age": "5 mo", "sex": "M"}, "age", "sex") == {"age": "5 months", "sex": "male"}
    assert enforce_schema({"age": "12/f"}, t, MEDICAL, normalize_age_and_sex=False)["age"] == "12/f"


def test_schema_block_matches_scripts_and_types():
    t = {"payee_name": "", "amount": "", "year": ""}
    text = add_schema_instructions("BASE", t, CHEQUE)
    assert "REQUIRED PREDEFINED SCHEMA" in text and "exactly the same spelling" in text
    assert "amount must be JSON integers" in text
    assert "Every value must be a string." in add_schema_instructions("BASE", {"a": ""}, [{"key": "a", "type": "string"}])
    assert "AGE AND SEX" in add_schema_instructions("B", {"name": "", "age": "", "sex": ""}, MEDICAL, True)


def test_render_prompt_appends_schema_only_with_schema():
    sb = render_prompt(prompt_for("schema_based"), {"a": ""}, [{"key": "a", "type": "string"}])
    assert "REQUIRED PREDEFINED SCHEMA" in sb
    sf = render_prompt(prompt_for("schema_free"), None)
    assert "REQUIRED PREDEFINED SCHEMA" not in sf and "snake_case" in sf


def test_image_to_base64_shrinks_and_jpeg():
    import base64, io
    data = image_to_base64(Image.new("RGB", (4000, 1000)), 1600)
    img = Image.open(io.BytesIO(base64.b64decode(data)))
    assert img.format == "JPEG" and max(img.size) == 1600


# ---- Qwen only ------------------------------------------------------------------

def test_qwen_only_schema_free():
    a = ScriptedAdapter(['```json\n{"name": "A", "address": {"city": "X"}}\n```'])
    out = QwenOnlyProcessor().process(IMG, ctx_for(make_task("schema_free", rules=[]), a))
    assert out.result == {"name": "A", "address.city": "X"} and out.status == "completed"


def test_qwen_only_schema_based_cheque():
    task = make_task("schema_based", CHEQUE)
    a = ScriptedAdapter(['{"payee_name": "Kumar Ravi Palli", "amount": "22,23,000/-", "year": "2016"}'])
    out = QwenOnlyProcessor().process(IMG, ctx_for(task, a))
    assert out.result == {"payee_name": "Kumar Ravi Palli", "amount": 2223000, "year": "2016"}
    assert "REQUIRED PREDEFINED SCHEMA" in a.prompts[0]


def test_qwen_only_parse_failure():
    out = QwenOnlyProcessor().process(IMG, ctx_for(make_task("schema_free"), ScriptedAdapter(["sorry"])))
    assert out.status == "parse_failed" and out.raw_text == "sorry"


# ---- real QwenAdapter against a local fake OpenAI-compatible server --------------

class _FakeVLLM(BaseHTTPRequestHandler):
    last_body = None
    bodies: list = []
    replies: list = []          # queued answers; default '{"name": "A"}'
    model_ids = ["qwen2.5-vl-7b"]

    def log_message(self, *args):
        pass

    def _send(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": m, "object": "model"} for m in _FakeVLLM.model_ids]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeVLLM.last_body = body
        _FakeVLLM.bodies.append(body)
        reply = _FakeVLLM.replies.pop(0) if _FakeVLLM.replies else '{"name": "A"}'
        self._send({"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": reply}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})


@pytest.fixture()
def fake_server():
    server = HTTPServer(("127.0.0.1", 0), _FakeVLLM)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    _FakeVLLM.bodies, _FakeVLLM.replies = [], []
    _FakeVLLM.model_ids = ["qwen2.5-vl-7b", "nuextract3", "glm-4.6v-flash"]
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def test_qwen_adapter_request_matches_scripts(fake_server):
    from models.base_adapter import ModelRequest
    from models.qwen_adapter import QwenAdapter
    a = QwenAdapter(base_url=fake_server, model_name="qwen2.5-vl-7b")
    assert a.check_available()[0]
    assert not QwenAdapter(base_url=fake_server, model_name="other").check_available()[0]
    resp = a.generate(ModelRequest(image=Image.new("RGB", (3000, 100)), prompt="Extract",
                                   params={"temperature": 0.0, "top_p": 1.0, "max_tokens": 3000}))
    assert resp.raw_text == '{"name": "A"}'
    body = _FakeVLLM.last_body
    content = body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Extract"}                  # prompt first
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert body["model"] == "qwen2.5-vl-7b" and body["max_tokens"] == 3000 and body["top_p"] == 1.0


# ---- NuExtract3 and GLM-4.6V-Flash against the fake server -----------------------

def test_nuextract_schema_based_sends_template_and_image_only(fake_server):
    from models.base_adapter import ModelRequest
    from models.nuextract_adapter import NuExtractAdapter
    a = NuExtractAdapter(base_url=fake_server, model_name="nuextract3")
    assert a.check_available()[0]
    _FakeVLLM.replies = ['```json\n{"payee_name": "Ravi", "amount": 2223000, "year": null}\n```']
    resp = a.generate(ModelRequest(image=IMG, prompt="ignored",
                                   params={"mode": "schema_based", "fields": CHEQUE, "user_prompt": ""}))
    assert json.loads(resp.raw_text)["payee_name"] == "Ravi"
    body = _FakeVLLM.last_body
    assert [c["type"] for c in body["messages"][0]["content"]] == ["image_url"]
    kw = body["chat_template_kwargs"]
    assert json.loads(kw["template"]) == {"payee_name": "verbatim-string", "amount": "integer",
                                          "year": "verbatim-string"}
    assert kw["enable_thinking"] is False and "instructions" not in kw


def test_nuextract_schema_free_generates_template_first(fake_server):
    from models.base_adapter import ModelRequest
    from models.nuextract_adapter import NuExtractAdapter
    from config.task_config import default_prompt
    _FakeVLLM.replies = ['{"name": "verbatim-string", "date": "date"}', '{"name": "A", "date": "2016-01-25"}']
    resp = NuExtractAdapter(base_url=fake_server).generate(ModelRequest(
        image=IMG, prompt="x", params={"mode": "schema_free", "user_prompt": default_prompt("schema_free") + "\nExtra rule"}))
    first, second = _FakeVLLM.bodies[-2:]
    assert first["chat_template_kwargs"]["mode"] == "template-generation"
    assert json.loads(second["chat_template_kwargs"]["template"]) == {"name": "verbatim-string", "date": "date"}
    assert "Extra rule" in second["chat_template_kwargs"]["instructions"]   # edited prompt -> instructions
    assert resp.metadata["generated_template"]["date"] == "date"


def test_nuextract_runs_through_team_processor(fake_server):
    from models.nuextract_adapter import NuExtractAdapter
    _FakeVLLM.replies = ['{"payee_name": "Ravi", "amount": "22,23,000", "year": null}']
    out = QwenOnlyProcessor().process(IMG, ctx_for(make_task("schema_based", CHEQUE),
                                                   NuExtractAdapter(base_url=fake_server)))
    assert out.result == {"payee_name": "Ravi", "amount": 2223000, "year": ""}


def test_glm_disables_thinking_and_cleans_output(fake_server):
    from models.base_adapter import ModelRequest
    from models.glm_adapter import GLMAdapter, clean_glm_output
    a = GLMAdapter(base_url=fake_server, model_name="glm-4.6v-flash")
    assert a.check_available()[0]
    _FakeVLLM.replies = ['<think>looking</think><|begin_of_box|>{"name": "A"}<|end_of_box|>']
    resp = a.generate(ModelRequest(image=IMG, prompt="Extract", params={}))
    assert resp.raw_text == '{"name": "A"}' and "raw_before_cleanup" in resp.metadata
    body = _FakeVLLM.last_body
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    content = body["messages"][0]["content"]
    assert content[0]["type"] == "image_url" and content[1] == {"type": "text", "text": "Extract"}
    assert clean_glm_output('reasoning cut</think>{"a": 1}') == '{"a": 1}'
