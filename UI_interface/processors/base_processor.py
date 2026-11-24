"""Processor interface shared by generic and custom processors."""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from config.task_config import TaskConfig
from models.base_adapter import BaseAdapter, ModelRequest, ModelResponse
from validation.rules import run_rules

SHARED_RULES_PATH = "prompts/shared_rules.txt"


@dataclass
class ProcessingContext:
    task: TaskConfig                             # mode, fields, template, schema, rules
    adapter: BaseAdapter
    model_name: str
    prompt: str                                  # edited in the UI, may contain placeholders
    params: dict[str, Any] = field(default_factory=dict)
    cancel_event: threading.Event = field(default_factory=threading.Event)

    @property
    def template(self) -> dict[str, Any] | None:
        return self.task.template

    @property
    def schema(self) -> dict[str, Any] | None:
        return self.task.schema


@dataclass
class ProcessOutput:
    status: str                        # completed | needs_review | parse_failed | failed
    result: dict[str, Any] | None
    raw_text: str
    flags: list[dict[str, str]] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)
    error: str | None = None
    latency_ms: float | None = None
    peak_memory_mb: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    rendered_prompt: str = ""


def load_shared_rules() -> str:
    from config.settings import BASE_DIR
    path = BASE_DIR / SHARED_RULES_PATH
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def render_prompt(prompt: str, template: dict[str, Any] | None, fields: list[dict[str, Any]] | None = None,
                  normalize_age_sex: bool = True) -> str:
    """Build the prompt sent to the model.

    With a schema (template is not None) the REQUIRED PREDEFINED SCHEMA block
    from the team's scripts is appended, unless the prompt places the template
    itself with a {template} placeholder. {fields} and {shared_rules} are also
    supported for custom prompts. Uses str.replace, so JSON braces are safe.
    """
    import json

    from processors.dat_extraction import add_schema_instructions
    from validation.schema_validator import fields_prompt_block
    out = prompt
    placed = "{template}" in out
    if placed:
        out = out.replace("{template}", json.dumps(template or {}, ensure_ascii=False))
    if "{fields}" in out:
        out = out.replace("{fields}", fields_prompt_block(fields or []))
    if "{shared_rules}" in out:
        out = out.replace("{shared_rules}", load_shared_rules())
    if template is not None and not placed:
        out = add_schema_instructions(out, template, fields, normalize_age_sex)
    return out.strip()


class BaseProcessor(ABC):
    name = "base"

    @abstractmethod
    def process(self, image: Image.Image, ctx: ProcessingContext) -> ProcessOutput:
        ...

    # helpers shared by subclasses
    def call_model(self, image: Image.Image, prompt: str, ctx: ProcessingContext,
                   json_schema: dict[str, Any] | None = None) -> ModelResponse:
        params = {**ctx.params, "mode": ctx.task.mode, "template": ctx.template, "fields": ctx.task.fields,
                  "user_prompt": ctx.prompt}
        return ctx.adapter.generate(ModelRequest(image=image, prompt=prompt, json_schema=json_schema,
                                                 params=params), ctx.cancel_event)

    @staticmethod
    def finalize(result: dict[str, Any], ctx: ProcessingContext, schema_errors: list[str] | None = None) -> tuple[str, list[dict[str, str]]]:
        flags = run_rules(result, ctx.task.rules)
        for err in schema_errors or []:
            flags.append({"rule": "schema", "field": err.split(":", 1)[0], "detail": err})
        return ("needs_review" if flags else "completed"), flags
