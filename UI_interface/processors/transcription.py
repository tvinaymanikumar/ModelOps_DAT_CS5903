"""Generic handwriting transcription."""
from __future__ import annotations

import re

from PIL import Image

from processors.base_processor import BaseProcessor, ProcessingContext, ProcessOutput, render_prompt

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def clean_transcription(raw: str) -> str:
    """Remove wrapping code fences and surrounding blank lines only.
    The raw response is kept untouched separately for evaluation."""
    text = _FENCE.sub("", raw.strip())
    lines = [ln.rstrip() for ln in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


class TranscriptionProcessor(BaseProcessor):
    name = "transcription"

    def process(self, image: Image.Image, ctx: ProcessingContext) -> ProcessOutput:
        prompt = render_prompt(ctx.prompt, None)
        resp = self.call_model(image, prompt, ctx)
        text = clean_transcription(resp.raw_text)
        result = {"transcription": text, "lines": text.splitlines()}
        status, flags = self.finalize(result, ctx)
        return ProcessOutput(status=status, result=result, raw_text=resp.raw_text, flags=flags,
                             latency_ms=resp.latency_ms, peak_memory_mb=resp.peak_memory_mb,
                             metadata=dict(resp.metadata), rendered_prompt=prompt)
