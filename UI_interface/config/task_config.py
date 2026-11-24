"""Task configuration: what to extract, independent of any dataset.

A task is defined at run time in the UI:

  mode = "schema_free"    extract every visible field as key-value pairs
  mode = "schema_based"   extract only the fields the user lists
  mode = "transcription"  transcribe handwriting to plain text

For schema-based tasks the user's field list generates both the simple
JSON template sent to the model and the formal JSON Schema used for
validation (and constrained decoding where the model supports it).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from config.settings import BASE_DIR

MODES = ["schema_free", "schema_based", "transcription"]
MODE_LABELS = {
    "schema_free": "Schema-free extraction",
    "schema_based": "Schema-based extraction",
    "transcription": "Handwriting transcription",
}
MODE_HELP = {
    "schema_free": "The model returns every visible field as key-value pairs. No field list needed.",
    "schema_based": "You list the fields to extract. The output always has exactly those keys.",
    "transcription": "The model transcribes handwritten text. Output is plain text.",
}
DEFAULT_PROMPT_FILES = {
    "schema_free": "prompts/schema_free.txt",
    "schema_based": "prompts/schema_based.txt",
    "transcription": "prompts/transcription.txt",
}


def default_prompt(mode: str) -> str:
    path = BASE_DIR / DEFAULT_PROMPT_FILES[mode]
    return path.read_text(encoding="utf-8") if path.exists() else ""


@dataclass
class TaskConfig:
    mode: str
    name: str = "Untitled task"
    fields: list[dict[str, Any]] = field(default_factory=list)   # schema-based only
    template: dict[str, Any] | None = None
    schema: dict[str, Any] | None = None
    strict_keys: bool = True              # drop keys that are not in the field list
    missing_value: str = ""
    flatten_separator: str = "."          # nested objects -> "a.b" (as in the team's flatten_json)
    duplicate_key_strategy: str = "suffix"  # schema-free: NAME, NAME_2 ... or "list"
    rules: list[str] = field(default_factory=list)   # validation rule specs, see validation/rules.py
    custom_processor: str | None = None   # optional "module:Class"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got '{self.mode}'")

    @property
    def label(self) -> str:
        return MODE_LABELS[self.mode]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})
