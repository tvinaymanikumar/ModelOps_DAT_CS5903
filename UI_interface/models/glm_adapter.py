"""GLM-4.6V-Flash adapter (self-hosted vLLM, OpenAI-compatible).

Configure in .env:

    GLM_BASE_URL=http://10.68.0.39:9007/v1
    GLM_MODEL_NAME=glm-4.6v-flash
    GLM_ENABLE_THINKING=false

GLM-4.6V is a hybrid reasoning model and vLLM enables thinking by default, so
thinking is switched off per request (chat_template_kwargs.enable_thinking).
If the server was started without --reasoning-parser glm45 the answer may still
contain <think>...</think> or <|begin_of_box|> markers; clean_glm_output()
removes them before the JSON is parsed. The untouched text is kept in metadata.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Any

from config.settings import env, env_bool
from models.base_adapter import BaseAdapter, ModelRequest, ModelResponse, NotConfiguredError
from models.dat_endpoints import (DEFAULT_MAX_TOKENS, DEFAULT_TIMEOUT, DEFAULT_TOP_P, MAX_IMAGE_SIDE,
                                  call_vision_model, image_to_base64, list_models, make_client)

_THINK = re.compile(r"<think>.*?</think>", re.S)
_BOX = re.compile(r"<\|/?(?:begin|end)_of_box\|>")


def clean_glm_output(text: str) -> str:
    """Drop reasoning and answer-box markers that GLM may put around the answer."""
    out = _THINK.sub("", text or "")
    if "</think>" in out:  # reasoning start was cut off, keep only the answer
        out = out.rsplit("</think>", 1)[1]
    return _BOX.sub("", out).strip()


class GLMAdapter(BaseAdapter):
    name = "GLM-4.6V-Flash"
    supports_guided_json = True
    description = "GLM-4.6V-Flash (9B) on the team's OpenAI-compatible endpoint."

    def __init__(self, base_url: str | None = None, model_name: str | None = None,
                 timeout_s: float = DEFAULT_TIMEOUT, **options: Any):
        super().__init__(**options)
        self.base_url = (base_url if base_url is not None else env("GLM_BASE_URL")).strip()
        self.model_name = model_name or env("GLM_MODEL_NAME", "glm-4.6v-flash")
        self.enable_thinking = env_bool("GLM_ENABLE_THINKING", False)
        self.timeout_s = float(timeout_s)
        self._client = None

    def check_available(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, "GLM: set GLM_BASE_URL in .env"
        ok, msg, ids = list_models(self.base_url)
        if not ok:
            return False, f"GLM {self.base_url}: {msg}"
        if ids and self.model_name not in ids:
            return False, f"GLM: model '{self.model_name}' not served (available: {', '.join(ids)})"
        return True, f"GLM: {msg}"

    def generate(self, request: ModelRequest, cancel_event: threading.Event | None = None) -> ModelResponse:
        if not self.base_url:
            raise NotConfiguredError("GLM-4.6V-Flash is not configured. Set GLM_BASE_URL in .env.")
        self.check_cancel(cancel_event)
        if self._client is None:
            self._client = make_client(self.base_url, self.timeout_s)
        p = request.params
        extra: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        if request.json_schema and p.get("guided_json"):
            extra["guided_json"] = request.json_schema
        start = time.perf_counter()
        raw = call_vision_model(
            self._client, self.model_name, request.prompt,
            image_to_base64(request.image, int(p.get("max_image_side", MAX_IMAGE_SIDE))),
            image_first=True,
            temperature=float(p.get("temperature", 0.0)),
            top_p=float(p.get("top_p", DEFAULT_TOP_P)),
            max_tokens=int(p.get("max_tokens", DEFAULT_MAX_TOKENS)),
            extra_body=extra,
        )
        text = clean_glm_output(raw)
        meta: dict[str, Any] = {"endpoint_model": self.model_name}
        if text != raw:
            meta["raw_before_cleanup"] = raw
        return ModelResponse(raw_text=text, latency_ms=(time.perf_counter() - start) * 1000, metadata=meta)
