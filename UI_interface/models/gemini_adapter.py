"""Gemini adapter (optional frontier baseline, EXTERNAL).

Images sent to this adapter leave the organization. The UI shows a warning and
requires confirmation, and ALLOW_EXTERNAL_MODELS=false hides it completely.
Use only for masked or non-sensitive documents.

Requires `pip install google-genai` and GEMINI_API_KEY in .env.
"""
from __future__ import annotations

import io
import threading
import time
from typing import Any

from config.settings import env
from models.base_adapter import (
    AdapterError,
    BaseAdapter,
    ModelRequest,
    ModelResponse,
    NotConfiguredError,
    TransientAdapterError,
)


class GeminiAdapter(BaseAdapter):
    name = "Gemini"
    is_external = True
    supports_guided_json = True
    description = "Hosted frontier model. Data leaves the organization."

    def __init__(self, **options: Any):
        super().__init__(**options)
        self.api_key = options.get("api_key", env("GEMINI_API_KEY"))
        self.model_name = options.get("model_name", env("GEMINI_MODEL_NAME", "gemini-2.5-flash"))

    def check_available(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "Gemini: set GEMINI_API_KEY in .env"
        try:
            import google.genai  # noqa: F401
        except ImportError:
            return False, "Gemini: install the SDK with `pip install google-genai`"
        return True, "Gemini: key present (external API)"

    def generate(self, request: ModelRequest, cancel_event: threading.Event | None = None) -> ModelResponse:
        ok, reason = self.check_available()
        if not ok:
            raise NotConfiguredError(reason)
        self.check_cancel(cancel_event)
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        buf = io.BytesIO()
        request.image.convert("RGB").save(buf, format="JPEG", quality=92)
        config_kwargs: dict[str, Any] = {"temperature": float(request.params.get("temperature", 0.0)),
                                         "max_output_tokens": int(request.params.get("max_tokens", 2048))}
        if request.json_schema and request.params.get("guided_json") and request.params.get("mode") != "transcription":
            config_kwargs["response_mime_type"] = "application/json"
        start = time.perf_counter()
        try:
            resp = client.models.generate_content(
                model=self.model_name,
                contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"), request.prompt],
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:  # SDK raises several exception types
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg or "UNAVAILABLE" in msg:
                raise TransientAdapterError(f"Gemini: {type(exc).__name__}") from exc
            raise AdapterError(f"Gemini: {type(exc).__name__}: {msg[:200]}") from exc
        latency = (time.perf_counter() - start) * 1000
        return ModelResponse(raw_text=resp.text or "", latency_ms=latency, metadata={"external": True})
