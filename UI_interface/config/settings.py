"""Central settings: paths and environment variables.

All credentials and endpoints come from `.env` (see `.env.example`).
Nothing in this module logs secret values.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)

PRESETS_DIR = BASE_DIR / "presets"
PROMPTS_DIR = BASE_DIR / "prompts"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

APP_TITLE = "Document Extraction"


def env(key: str, default: str = "") -> str:
    """Read an environment variable, stripped. Empty string when unset."""
    return (os.getenv(key, default) or "").strip()


def env_bool(key: str, default: bool = False) -> bool:
    value = env(key, "true" if default else "false").lower()
    return value in {"1", "true", "yes", "on"}


def env_float(key: str, default: float) -> float:
    try:
        return float(env(key, str(default)))
    except ValueError:
        return default


def env_int(key: str, default: int) -> int:
    try:
        return int(env(key, str(default)))
    except ValueError:
        return default


ALLOW_EXTERNAL_MODELS = env_bool("ALLOW_EXTERNAL_MODELS", False)
REQUEST_TIMEOUT_S = env_float("REQUEST_TIMEOUT_S", 180.0)
DEFAULT_PDF_DPI = env_int("PDF_DPI", 200)

# Default per-model rate limits (requests per minute). 0 = unlimited.
DEFAULT_RPM = {
    "Mock Model": 0,
    "Qwen2.5-VL-7B": 0,
    "NuExtract3": 0,
    "GLM-4.6V-Flash": 0,
    "Gemini": 15,
}

SUPPORTED_UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "gif", "pdf"]

for _d in (UPLOADS_DIR, OUTPUTS_DIR, PRESETS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
