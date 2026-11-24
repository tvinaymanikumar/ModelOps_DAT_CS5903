"""Model registry: display name -> ModelSpec(adapter class, supported modes,
optional processor).

Models are independent of any dataset. Each model declares which task modes
it can run; the sidebar only lists models that support the chosen mode.

Registered models: Mock Model, Qwen2.5-VL-7B, NuExtract3, GLM-4.6V-Flash and
Gemini (external, hidden unless ALLOW_EXTERNAL_MODELS=true). The three
self-hosted models use the team's Qwen-script processor for both schema options
(processors/custom/qwen_scripts.py); transcription uses the generic processor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config.settings import ALLOW_EXTERNAL_MODELS
from models.base_adapter import BaseAdapter
from models.gemini_adapter import GeminiAdapter
from models.glm_adapter import GLMAdapter
from models.mock_adapter import MockAdapter
from models.nuextract_adapter import NuExtractAdapter
from models.qwen_adapter import QwenAdapter

ALL_MODES = ["schema_free", "schema_based", "transcription"]
EXTRACTION_MODES = ["schema_free", "schema_based"]


@dataclass
class ModelSpec:
    adapter: type[BaseAdapter]
    modes: list[str] = field(default_factory=lambda: list(ALL_MODES))
    processors: dict[str, str] = field(default_factory=dict)  # mode -> "module:Class" (else generic)

    @property
    def processor(self) -> str | None:  # backwards-compatible single value
        return next(iter(self.processors.values()), None)


TEAM_PROCESSOR = "processors.custom.qwen_scripts:QwenOnlyProcessor"
TEAM = {"schema_free": TEAM_PROCESSOR, "schema_based": TEAM_PROCESSOR}


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "Mock Model": ModelSpec(MockAdapter),
    "Qwen2.5-VL-7B": ModelSpec(QwenAdapter, processors=TEAM),
    "NuExtract3": ModelSpec(NuExtractAdapter, processors=TEAM),
    "GLM-4.6V-Flash": ModelSpec(GLMAdapter, processors=TEAM),
    "Gemini": ModelSpec(GeminiAdapter),
}


def is_external(model_name: str) -> bool:
    spec = MODEL_REGISTRY.get(model_name)
    return bool(spec and spec.adapter.is_external)


def get_adapter(model_name: str, **options: Any) -> BaseAdapter:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'. Registered: {sorted(MODEL_REGISTRY)}")
    if is_external(model_name) and not ALLOW_EXTERNAL_MODELS:
        raise PermissionError(f"{model_name} is an external API and ALLOW_EXTERNAL_MODELS=false")
    return MODEL_REGISTRY[model_name].adapter(**options)


def models_for_mode(mode: str) -> list[str]:
    """Registered models that support `mode` and are allowed by the privacy policy."""
    return [name for name, spec in MODEL_REGISTRY.items()
            if mode in spec.modes and (ALLOW_EXTERNAL_MODELS or not spec.adapter.is_external)]


def availability(model_name: str) -> tuple[bool, str]:
    try:
        adapter = get_adapter(model_name)
    except Exception as exc:
        return False, str(exc)
    return adapter.check_available()
