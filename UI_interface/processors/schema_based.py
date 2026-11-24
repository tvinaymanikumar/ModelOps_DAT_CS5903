"""Generic schema-based extraction: exactly the fields the user listed.
Needs only the field list (template + schema, generated in the UI) and a prompt."""
from __future__ import annotations

from typing import Any

from PIL import Image

from processors.base_processor import BaseProcessor, ProcessingContext, ProcessOutput, render_prompt
from validation.json_parser import flatten, parse_json_object
from validation.schema_validator import coerce_types, complete_template, validate


def postprocess_schema_result(obj: dict[str, Any], ctx: ProcessingContext) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Template completion, strict keys, type coercion and schema validation.
    Shared with custom processors."""
    meta: dict[str, Any] = {}
    template = ctx.template or {}
    flat = flatten(obj, " - ") if any(isinstance(v, dict) for v in obj.values()) else obj
    completed, missing, extra = complete_template(flat, template, ctx.task.strict_keys, ctx.task.missing_value)
    if missing:
        meta["missing_keys_added"] = missing
    if extra:
        meta["extra_keys"] = {k: flat[k] for k in extra}
    coerced, changed = coerce_types(completed, ctx.schema)
    if changed:
        meta["coerced_fields"] = changed
    errors = validate(coerced, ctx.schema) if ctx.task.strict_keys else []
    return coerced, errors, meta


class SchemaBasedProcessor(BaseProcessor):
    name = "schema_based"

    def process(self, image: Image.Image, ctx: ProcessingContext) -> ProcessOutput:
        prompt = render_prompt(ctx.prompt, ctx.template, ctx.task.fields,
                               normalize_age_sex=bool(ctx.params.get("normalize_age_sex", True)))
        use_schema = ctx.schema if ctx.params.get("guided_json") and ctx.adapter.supports_guided_json else None
        resp = self.call_model(image, prompt, ctx, json_schema=use_schema)
        meta = dict(resp.metadata)
        if resp.parsed is not None:
            obj, err = resp.parsed, None
        else:
            obj, err, renamed = parse_json_object(resp.raw_text, "suffix")
            if renamed:
                meta["duplicate_keys_renamed"] = sorted(set(renamed))
        if obj is None:
            return ProcessOutput(status="parse_failed", result=None, raw_text=resp.raw_text, error=err,
                                 latency_ms=resp.latency_ms, peak_memory_mb=resp.peak_memory_mb,
                                 metadata=meta, rendered_prompt=prompt)
        result, schema_errors, post_meta = postprocess_schema_result(obj, ctx)
        meta.update(post_meta)
        status, flags = self.finalize(result, ctx, schema_errors)
        return ProcessOutput(status=status, result=result, raw_text=resp.raw_text, flags=flags,
                             schema_errors=schema_errors, latency_ms=resp.latency_ms,
                             peak_memory_mb=resp.peak_memory_mb, metadata=meta, rendered_prompt=prompt)
