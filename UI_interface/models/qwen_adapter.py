"""Qwen2.5-VL-7B adapter using the team's endpoint code (models/dat_endpoints.py).

Defaults match the scripts: prompt first, then a JPEG image (max side 1600,
quality 95); temperature 0.0, top_p 1.0, max_tokens 3000. Configure in .env:

    QWEN_BASE_URL=http://10.68.0.39:9005/v1
    QWEN_MODEL_NAME=qwen2.5-vl-7b
"""
from __future__ import annotations

import threading
import time
from typing import Any

from config.settings import env
from models.base_adapter import BaseAdapter, ModelRequest, ModelResponse, NotConfiguredError
from models.dat_endpoints import (DEFAULT_MAX_TOKENS, DEFAULT_TIMEOUT, DEFAULT_TOP_P, MAX_IMAGE_SIDE,
                                  call_vision_model, image_to_base64, list_models, make_client)


class QwenAdapter(BaseAdapter):
    name = "Qwen2.5-VL-7B"
    supports_guided_json = True  # vLLM guided_json; off by default, as in the scripts
    description = "Qwen2.5-VL-7B on the team's OpenAI-compatible endpoint."

    def __init__(self, base_url: str | None = None, model_name: str | None = None,
                 timeout_s: float = DEFAULT_TIMEOUT, **options: Any):
        super().__init__(**options)
        self.base_url = (base_url if base_url is not None else env("QWEN_BASE_URL")).strip()
        self.model_name = model_name or env("QWEN_MODEL_NAME", "qwen2.5-vl-7b")
        self.timeout_s = float(timeout_s)
        self._client = None

    def check_available(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, "Qwen2.5-VL: set QWEN_BASE_URL in .env"
        ok, msg, ids = list_models(self.base_url)
        if not ok:
            return False, f"Qwen2.5-VL {self.base_url}: {msg}"
        if ids and self.model_name not in ids:
            return False, f"Qwen2.5-VL: model '{self.model_name}' not served (available: {', '.join(ids)})"
        return True, f"Qwen2.5-VL: {msg}"

    def generate(self, request: ModelRequest, cancel_event: threading.Event | None = None) -> ModelResponse:
        if not self.base_url:
            raise NotConfiguredError("Qwen2.5-VL is not configured. Set QWEN_BASE_URL in .env.")
        self.check_cancel(cancel_event)
        if self._client is None:
            self._client = make_client(self.base_url, self.timeout_s)
        p = request.params
        extra = None
        if request.json_schema and p.get("guided_json"):
            extra = {"guided_json": request.json_schema}
        start = time.perf_counter()
        text = call_vision_model(
            self._client, self.model_name, request.prompt,
            image_to_base64(request.image, int(p.get("max_image_side", MAX_IMAGE_SIDE))),
            image_first=False,
            temperature=float(p.get("temperature", 0.0)),
            top_p=float(p.get("top_p", DEFAULT_TOP_P)),
            max_tokens=int(p.get("max_tokens", DEFAULT_MAX_TOKENS)),
            extra_body=extra,
        )
        return ModelResponse(raw_text=text, latency_ms=(time.perf_counter() - start) * 1000,
                             metadata={"endpoint_model": self.model_name})
