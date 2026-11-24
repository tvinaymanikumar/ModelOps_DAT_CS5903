"""Resolve (task mode, model, optional processor choice) -> processor + adapter.

Priority:
 1. the custom processor chosen in the sidebar (config/processor_registry.py)
 2. task.custom_processor, if set programmatically
 3. the model's processor for this mode (the team's Qwen-script processor for Qwen, NuExtract3 and GLM)
 4. the generic processor for the mode
The router holds no prompts, keys or extraction logic.
"""
from __future__ import annotations

import importlib
from typing import Any

from config.model_registry import MODEL_REGISTRY, get_adapter
from config.processor_registry import CUSTOM_PROCESSORS
from config.task_config import MODES
from models.base_adapter import BaseAdapter
from processors.base_processor import BaseProcessor
from processors.schema_based import SchemaBasedProcessor
from processors.schema_free import SchemaFreeProcessor
from processors.transcription import TranscriptionProcessor

GENERIC_PROCESSORS: dict[str, type[BaseProcessor]] = {
    "schema_free": SchemaFreeProcessor,
    "schema_based": SchemaBasedProcessor,
    "transcription": TranscriptionProcessor,
}


class RoutingError(RuntimeError):
    pass


def _load_class(path: str) -> type[BaseProcessor]:
    if ":" not in path:
        raise RoutingError(f"Custom processor '{path}' must look like 'module.path:ClassName'")
    module_name, cls_name = path.split(":", 1)
    try:
        cls = getattr(importlib.import_module(module_name), cls_name)
    except (ImportError, AttributeError) as exc:
        raise RoutingError(f"Cannot load custom processor '{path}': {exc}") from exc
    if not (isinstance(cls, type) and issubclass(cls, BaseProcessor)):
        raise RoutingError(f"'{path}' is not a BaseProcessor subclass")
    return cls


def resolve_processor_path(mode: str, model_name: str, processor_choice: str | None = None,
                           task_custom: str | None = None) -> str:
    if mode not in MODES:
        raise RoutingError(f"Unknown mode '{mode}'")
    spec = MODEL_REGISTRY.get(model_name)
    if spec is None:
        raise RoutingError(f"{model_name} has no registered adapter")
    if mode not in spec.modes:
        raise RoutingError(f"{model_name} does not support {mode.replace('_', '-')} tasks")
    if processor_choice:
        entry = CUSTOM_PROCESSORS.get(processor_choice)
        if entry is None:
            raise RoutingError(f"Unknown processor '{processor_choice}'")
        if mode not in entry.get("modes", []):
            raise RoutingError(f"Processor '{processor_choice}' does not support {mode} tasks")
        return entry["path"]
    if task_custom:
        return task_custom
    if mode in spec.processors:
        return spec.processors[mode]
    return f"generic:{mode}"


def route(mode: str, model_name: str, processor_choice: str | None = None,
          adapter_options: dict[str, Any] | None = None, adapter: BaseAdapter | None = None,
          task_custom: str | None = None) -> tuple[BaseProcessor, BaseAdapter, str]:
    path = resolve_processor_path(mode, model_name, processor_choice, task_custom)
    processor = GENERIC_PROCESSORS[mode]() if path.startswith("generic:") else _load_class(path)()
    adapter = adapter or get_adapter(model_name, **(adapter_options or {}))
    return processor, adapter, path
