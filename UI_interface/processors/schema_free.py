"""Generic schema-free extraction: every visible field as key-value pairs."""
from __future__ import annotations

from PIL import Image

from processors.base_processor import BaseProcessor, ProcessingContext, ProcessOutput, render_prompt
from validation.json_parser import flatten, parse_json_object


class SchemaFreeProcessor(BaseProcessor):
    name = "schema_free"

    def process(self, image: Image.Image, ctx: ProcessingContext) -> ProcessOutput:
        prompt = render_prompt(ctx.prompt, None)
        resp = self.call_model(image, prompt, ctx)
        meta = dict(resp.metadata)
        if resp.parsed is not None:
            obj, err, renamed = resp.parsed, None, []
        else:
            obj, err, renamed = parse_json_object(resp.raw_text, ctx.task.duplicate_key_strategy)
        if obj is None:
            return ProcessOutput(status="parse_failed", result=None, raw_text=resp.raw_text, error=err,
                                 latency_ms=resp.latency_ms, peak_memory_mb=resp.peak_memory_mb,
                                 metadata=meta, rendered_prompt=prompt)
        result = flatten(obj, ctx.task.flatten_separator)
        if renamed:
            meta["duplicate_keys_renamed"] = sorted(set(renamed))
        status, flags = self.finalize(result, ctx)
        return ProcessOutput(status=status, result=result, raw_text=resp.raw_text, flags=flags,
                             latency_ms=resp.latency_ms, peak_memory_mb=resp.peak_memory_mb,
                             metadata=meta, rendered_prompt=prompt)
