"""Extraction processor built from the team's Qwen scripts ("Approach 2").

QwenOnlyProcessor
    image -> model (extraction prompt [+ schema block]) -> parse -> flatten -> enforce schema

Used for "No schema" (schema_free) and "With schema" (schema_based) by
Qwen2.5-VL-7B, GLM-4.6V-Flash and NuExtract3 (config/model_registry.py).
NuExtract3 builds its own template from the field list inside its adapter, so
the same parsing / schema enforcement applies to all three models.
"""
from __future__ import annotations

from typing import Any

from PIL import Image

from processors.base_processor import BaseProcessor, ProcessingContext, ProcessOutput, render_prompt
from processors.dat_extraction import enforce_schema, flatten_json, parse_json_response
from validation.schema_validator import validate

class QwenOnlyProcessor(BaseProcessor):
    name = "qwen_only"

    def _normalize_flag(self, ctx: ProcessingContext) -> bool:
        return bool(ctx.params.get("normalize_age_sex", True))

    def extract(self, image: Image.Image, ctx: ProcessingContext) -> tuple[dict[str, Any] | None, str, str | None, float | None, str]:
        """Run Approach 2. Returns (fields or None, raw text, parse error, latency, prompt)."""
        schema_based = ctx.task.mode == "schema_based"
        prompt = render_prompt(ctx.prompt, ctx.template if schema_based else None, ctx.task.fields,
                               normalize_age_sex=self._normalize_flag(ctx))
        use_schema = ctx.schema if (schema_based and ctx.params.get("guided_json")) else None
        resp = self.call_model(image, prompt, ctx, json_schema=use_schema)
        self._last_meta = dict(resp.metadata or {})
        try:
            parsed = resp.parsed if resp.parsed is not None else parse_json_response(resp.raw_text)
        except (ValueError, TypeError) as exc:
            return None, resp.raw_text, str(exc), resp.latency_ms, prompt
        fields = flatten_json(parsed, separator=ctx.task.flatten_separator)
        if schema_based:
            fields = enforce_schema(fields, ctx.template, ctx.task.fields, self._normalize_flag(ctx))
        return fields, resp.raw_text, None, resp.latency_ms, prompt

    def _finish(self, fields: dict[str, Any], raw: str, ctx: ProcessingContext, latency: float | None,
                prompt: str, meta: dict[str, Any], extra_flags: list[dict[str, str]] | None = None) -> ProcessOutput:
        errors = validate(fields, ctx.schema) if ctx.task.mode == "schema_based" else []
        status, flags = self.finalize(fields, ctx, errors)
        flags = list(extra_flags or []) + flags
        if flags:
            status = "needs_review"
        return ProcessOutput(status=status, result=fields, raw_text=raw, flags=flags, schema_errors=errors,
                             latency_ms=latency, metadata=meta, rendered_prompt=prompt)

    def process(self, image: Image.Image, ctx: ProcessingContext) -> ProcessOutput:
        fields, raw, err, latency, prompt = self.extract(image, ctx)
        meta = {"approach": "qwen_only", **getattr(self, "_last_meta", {})}
        if fields is None:
            return ProcessOutput(status="parse_failed", result=None, raw_text=raw, error=err,
                                 latency_ms=latency, rendered_prompt=prompt, metadata=meta)
        return self._finish(fields, raw, ctx, latency, prompt, meta)
