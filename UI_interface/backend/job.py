"""Job and item state. Shared between the UI thread and the worker thread;
all mutations go through Job.lock."""
from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from config.settings import OUTPUTS_DIR, UPLOADS_DIR
from config.task_config import TaskConfig


class ItemStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    PARSE_FAILED = "parse_failed"
    FAILED = "failed"
    CANCELLED = "cancelled"


SUCCESS = {ItemStatus.COMPLETED, ItemStatus.NEEDS_REVIEW}
RETRYABLE = {ItemStatus.FAILED, ItemStatus.PARSE_FAILED, ItemStatus.CANCELLED}


@dataclass
class Item:
    item_id: str
    source_name: str          # uploaded file name
    page: int                 # 1-based page/frame
    image_path: str           # saved PNG inside the job's upload folder
    status: ItemStatus = ItemStatus.PENDING
    error: str | None = None
    attempts: int = 0
    latency_ms: float | None = None
    flags: list[dict[str, str]] = field(default_factory=list)
    output_stem: str = ""
    multipage: bool = False
    envelope: dict[str, Any] | None = None

    @property
    def display_name(self) -> str:
        return f"{self.source_name} (page {self.page})" if self.multipage else self.source_name


@dataclass
class Job:
    task: TaskConfig
    model_name: str
    prompt: str
    params: dict[str, Any]
    output_formats: list[str]
    processor_choice: str | None = None   # name from config/processor_registry.py, or None
    job_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    items: list[Item] = field(default_factory=list)
    finished: bool = False
    current_item_id: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def upload_dir(self) -> Path:
        return UPLOADS_DIR / self.job_id

    @property
    def output_dir(self) -> Path:
        return OUTPUTS_DIR / self.job_id

    def counts(self) -> dict[str, int]:
        with self.lock:
            c = {s.value: 0 for s in ItemStatus}
            for it in self.items:
                c[it.status.value] += 1
            return c

    def snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            return [{"item_id": it.item_id, "name": it.display_name, "status": it.status.value,
                     "flags": len(it.flags), "latency_ms": it.latency_ms, "attempts": it.attempts,
                     "error": it.error or ""} for it in self.items]

    def get_item(self, item_id: str) -> Item | None:
        return next((it for it in self.items if it.item_id == item_id), None)

    @property
    def mode(self) -> str:
        return self.task.mode

    def settings_dict(self) -> dict[str, Any]:
        return {"job_id": self.job_id, "created_at": self.created_at, "task_name": self.task.name,
                "mode": self.task.mode, "model": self.model_name, "processor_choice": self.processor_choice,
                "prompt": self.prompt, "task": self.task.to_dict(), "params": self.params,
                "output_formats": self.output_formats}
