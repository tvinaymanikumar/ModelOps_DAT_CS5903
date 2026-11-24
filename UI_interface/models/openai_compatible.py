"""Shared client for OpenAI-compatible chat-completions endpoints.

Works with vLLM (`vllm serve ...`), LMDeploy, SGLang, TGI and similar servers
that expose `/v1/chat/completions` with image_url content parts.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import requests

from config.settings import REQUEST_TIMEOUT_S
from models.base_adapter import (
    AdapterError,
    BaseAdapter,
    ModelRequest,
    ModelResponse,
    NotConfiguredError,
    TransientAdapterError,
)


class OpenAICompatibleAdapter(BaseAdapter):
    name = "OpenAI-compatible"
    supports_guided_json = True

    def __init__(self, base_url: str = "", model_name: str = "", api_key: str = "",
                 timeout_s: float = REQUEST_TIMEOUT_S, image_format: str = "PNG",
                 guided_json_style: str = "vllm", **options: Any):
        super().__init__(**options)
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.image_format = image_format
        self.guided_json_style = guided_json_style  # "vllm" | "response_format" | "none"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def check_available(self) -> tuple[bool, str]:
        if not self.base_url or not self.model_name:
            return False, f"{self.name}: set base URL and model name in .env"
        try:
            r = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=5)
            if r.status_code == 200:
                return True, f"{self.name}: endpoint reachable"
            return False, f"{self.name}: endpoint returned HTTP {r.status_code}"
        except requests.RequestException as exc:
            return False, f"{self.name}: endpoint unreachable ({type(exc).__name__})"

    def build_payload(self, request: ModelRequest) -> dict[str, Any]:
        p = request.params
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self.data_url(request.image, self.image_format)}},
                    {"type": "text", "text": request.prompt},
                ],
            }],
            "temperature": float(p.get("temperature", 0.0)),
            "max_tokens": int(p.get("max_tokens", 2048)),
        }
        if request.json_schema and p.get("guided_json", False) and self.supports_guided_json:
            if self.guided_json_style == "vllm":
                payload["guided_json"] = request.json_schema
            elif self.guided_json_style == "response_format":
                payload["response_format"] = {"type": "json_schema",
                                              "json_schema": {"name": "extraction", "schema": request.json_schema}}
        return payload

    def generate(self, request: ModelRequest, cancel_event: threading.Event | None = None) -> ModelResponse:
        if not self.base_url or not self.model_name:
            raise NotConfiguredError(f"{self.name} is not configured. Set its base URL and model name in .env.")
        self.check_cancel(cancel_event)
        payload = self.build_payload(request)
        start = time.perf_counter()
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=payload,
                              headers=self._headers(), timeout=self.timeout_s)
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise TransientAdapterError(f"{self.name}: {type(exc).__name__}") from exc
        latency = (time.perf_counter() - start) * 1000
        if r.status_code == 429 or r.status_code >= 500:
            retry_after = r.headers.get("Retry-After")
            raise TransientAdapterError(f"{self.name}: HTTP {r.status_code}",
                                        retry_after=float(retry_after) if retry_after and retry_after.isdigit() else None)
        if r.status_code >= 400:
            raise AdapterError(f"{self.name}: HTTP {r.status_code} {r.text[:200]}")
        try:
            data = r.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError) as exc:
            raise AdapterError(f"{self.name}: unexpected response format") from exc
        if isinstance(text, list):  # some servers return content parts
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        return ModelResponse(raw_text=text, latency_ms=latency, usage=data.get("usage"),
                             peak_memory_mb=None, metadata={"endpoint_model": data.get("model")})
