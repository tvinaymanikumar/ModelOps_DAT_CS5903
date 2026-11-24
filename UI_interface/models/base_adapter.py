"""Common adapter interface.

An adapter handles model-specific communication only: endpoint, auth, request
format, image encoding, parameters and response extraction. It never contains
task-specific extraction logic (that lives in processors).
"""
from __future__ import annotations

import base64
import io
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image


class AdapterError(RuntimeError):
    """Non-retryable model error (bad request, auth failure, bad response)."""


class NotConfiguredError(AdapterError):
    """Raised when an adapter's endpoint, key or script is not configured."""


class TransientAdapterError(RuntimeError):
    """Retryable error: timeout, connection reset, HTTP 429 or 5xx."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class CancelledError(RuntimeError):
    """Raised when processing is cancelled before a request was sent."""


@dataclass
class ModelRequest:
    image: Image.Image
    prompt: str
    json_schema: dict[str, Any] | None = None  # only used if adapter.supports_guided_json
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    raw_text: str                          # always preserved, never altered
    parsed: dict[str, Any] | None = None   # set only when the model returns structured output natively
    latency_ms: float | None = None
    peak_memory_mb: float | None = None    # None when unknown, never 0
    usage: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAdapter(ABC):
    name: str = "base"
    is_external: bool = False          # True for hosted third-party APIs (data leaves the organization)
    supports_guided_json: bool = False
    description: str = ""

    def __init__(self, **options: Any):
        self.options = options

    @abstractmethod
    def check_available(self) -> tuple[bool, str]:
        """Return (available, human-readable reason). Must not send document data."""

    @abstractmethod
    def generate(self, request: ModelRequest, cancel_event: threading.Event | None = None) -> ModelResponse:
        """Run the model on one image."""

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def encode_image(image: Image.Image, fmt: str = "PNG") -> str:
        buf = io.BytesIO()
        img = image
        if fmt.upper() == "JPEG" and img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @classmethod
    def data_url(cls, image: Image.Image, fmt: str = "PNG") -> str:
        mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
        return f"data:{mime};base64,{cls.encode_image(image, fmt)}"

    @staticmethod
    def check_cancel(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("Cancelled before the request was sent")
