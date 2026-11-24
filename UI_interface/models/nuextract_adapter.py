"""NuExtract3 adapter (self-hosted vLLM, OpenAI-compatible).

Configure in .env:

    NUEXTRACT_BASE_URL=http://10.68.0.39:9003/v1
    NUEXTRACT_MODEL_NAME=nuextract3
    NUEXTRACT_ENABLE_THINKING=false

NuExtract3 is not prompted like a chat model. The message holds only the image;
what to extract goes in chat_template_kwargs:

    With schema   template = the user's fields as a NuExtract template
                  ({"payee_name": "verbatim-string", "amount": "number", ...})
    No schema     two calls: mode="template-generation" on the image gives a
                  template, which is then used for the extraction call
    Transcription mode="content" (plain text of the page)

The prompt from the "Edit the prompt" box is sent as NuExtract `instructions`
only when you changed it from the default (the default prompt is written for
chat models and would only confuse NuExtract). Field hints are always sent as
instructions.
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

from config.settings import env, env_bool
from models.base_adapter import AdapterError, BaseAdapter, ModelRequest, ModelResponse, NotConfiguredError
from models.dat_endpoints import (DEFAULT_MAX_TOKENS, DEFAULT_TIMEOUT, DEFAULT_TOP_P, MAX_IMAGE_SIDE,
                                  call_vision_model, image_to_base64, list_models, make_client)

# app field type -> NuExtract template type
TYPE_MAP: dict[str, Any] = {
    "string": "verbatim-string",
    "integer": "integer",
    "number": "number",
    "date": "date",
    "boolean": ["true", "false"],
}


def nuextract_template(fields: list[dict[str, Any]]) -> dict[str, Any]:
    return {f["key"]: TYPE_MAP.get(f.get("type", "string"), "verbatim-string") for f in fields}


def _strip(text: str) -> str:
    """Remove a reasoning block and Markdown fences around the answer."""
    out = text or ""
    if "</think>" in out:
        out = out.rsplit("</think>", 1)[1]
    out = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", out.strip(), flags=re.I)
    return out.strip()


class NuExtractAdapter(BaseAdapter):
    name = "NuExtract3"
    supports_guided_json = False  # the template already fixes the output structure
    description = "NuMind NuExtract3 (4B) structured-extraction model on the team's endpoint."

    def __init__(self, base_url: str | None = None, model_name: str | None = None,
                 timeout_s: float = DEFAULT_TIMEOUT, **options: Any):
        super().__init__(**options)
        self.base_url = (base_url if base_url is not None else env("NUEXTRACT_BASE_URL")).strip()
        self.model_name = model_name or env("NUEXTRACT_MODEL_NAME", "nuextract3")
        self.enable_thinking = env_bool("NUEXTRACT_ENABLE_THINKING", False)
        self.timeout_s = float(timeout_s)
        self._client = None

    def check_available(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, "NuExtract3: set NUEXTRACT_BASE_URL in .env"
        ok, msg, ids = list_models(self.base_url)
        if not ok:
            return False, f"NuExtract3 {self.base_url}: {msg}"
        if ids and self.model_name not in ids:
            return False, f"NuExtract3: model '{self.model_name}' not served (available: {', '.join(ids)})"
        return True, f"NuExtract3: {msg}"

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _instructions(params: dict[str, Any]) -> str:
        from config.task_config import default_prompt
        parts = []
        mode = params.get("mode", "schema_free")
        user_prompt = (params.get("user_prompt") or "").strip()
        if user_prompt and user_prompt != default_prompt(mode).strip():
            parts.append(user_prompt)
        hints = [f"- {f['key']}: {f['description']}" for f in params.get("fields") or [] if f.get("description")]
        if hints:
            parts.append("Field hints:\n" + "\n".join(hints))
        return "\n\n".join(parts)

    def _call(self, image_b64: str, kwargs: dict[str, Any], params: dict[str, Any]) -> str:
        return call_vision_model(
            self._client, self.model_name, None, image_b64,
            temperature=float(params.get("temperature", 0.0)),
            top_p=float(params.get("top_p", DEFAULT_TOP_P)),
            max_tokens=int(params.get("max_tokens", DEFAULT_MAX_TOKENS)),
            extra_body={"chat_template_kwargs": {**kwargs, "enable_thinking": self.enable_thinking}},
        )

    # ---- main ------------------------------------------------------------
    def generate(self, request: ModelRequest, cancel_event: threading.Event | None = None) -> ModelResponse:
        if not self.base_url:
            raise NotConfiguredError("NuExtract3 is not configured. Set NUEXTRACT_BASE_URL in .env.")
        self.check_cancel(cancel_event)
        if self._client is None:
            self._client = make_client(self.base_url, self.timeout_s)
        p = request.params
        mode = p.get("mode", "schema_free")
        image_b64 = image_to_base64(request.image, int(p.get("max_image_side", MAX_IMAGE_SIDE)))
        meta: dict[str, Any] = {"endpoint_model": self.model_name}
        start = time.perf_counter()

        if mode == "transcription":
            text = _strip(self._call(image_b64, {"mode": "content"}, p))
            return ModelResponse(raw_text=text, latency_ms=(time.perf_counter() - start) * 1000, metadata=meta)

        if mode == "schema_based" and p.get("fields"):
            template = nuextract_template(p["fields"])
        else:  # schema-free: let NuExtract propose a template for this document first
            gen = _strip(self._call(image_b64, {"mode": "template-generation"}, p))
            try:
                template = json.loads(gen)
            except ValueError as exc:
                raise AdapterError(f"NuExtract3: template generation did not return JSON: {gen[:200]}") from exc
            if not isinstance(template, dict) or not template:
                raise AdapterError("NuExtract3: template generation returned an empty template")
            meta["generated_template"] = template
            self.check_cancel(cancel_event)

        kwargs: dict[str, Any] = {"template": json.dumps(template, indent=4, ensure_ascii=False)}
        instructions = self._instructions(p)
        if instructions:
            kwargs["instructions"] = instructions
            meta["instructions"] = instructions
        text = _strip(self._call(image_b64, kwargs, p))
        return ModelResponse(raw_text=text, latency_ms=(time.perf_counter() - start) * 1000, metadata=meta)
