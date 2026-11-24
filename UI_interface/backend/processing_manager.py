"""One background worker per Streamlit session.

Stored in st.session_state, so reruns reuse the same manager and can never
start a duplicate worker while one is alive.
"""
from __future__ import annotations

import threading
from typing import Any

from backend.batch_processor import BatchProcessor
from backend.job import Item, ItemStatus, Job, RETRYABLE
from backend.output_manager import OutputManager
from backend.rate_limiter import RateLimiter
from backend.router import route
from models.base_adapter import BaseAdapter


class ProcessingManager:
    def __init__(self) -> None:
        self.job: Job | None = None
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self.last_error: str | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def stop_requested(self) -> bool:
        return self._cancel.is_set() and self.is_running

    def set_job(self, job: Job) -> None:
        if self.is_running:
            raise RuntimeError("A job is already running")
        self.job = job

    def start(self, rpm: float = 0, min_delay_s: float = 0, max_retries: int = 2,
              adapter_options: dict[str, Any] | None = None, adapter: BaseAdapter | None = None) -> bool:
        with self._lock:
            if self.is_running or self.job is None:
                return False
            job = self.job
            processor, adapter, path = route(job.task.mode, job.model_name, job.processor_choice,
                                             adapter_options, adapter, job.task.custom_processor)
            job.params["processor"] = path
            with job.lock:
                job.finished = False
            self._cancel = threading.Event()
            batch = BatchProcessor(job, processor, adapter, RateLimiter(rpm, min_delay_s), self._cancel,
                                   OutputManager(job), max_retries=max_retries)
            self._thread = threading.Thread(target=batch.run, name=f"job-{job.job_id}", daemon=True)
            self._thread.start()
            return True

    def stop(self) -> None:
        self._cancel.set()

    def clear_queue(self) -> int:
        """Cancel pending items without touching finished ones."""
        if self.job is None:
            return 0
        n = 0
        with self.job.lock:
            for it in self.job.items:
                if it.status == ItemStatus.PENDING:
                    it.status = ItemStatus.CANCELLED
                    n += 1
        return n

    def reset_retryable(self) -> int:
        """Mark failed / parse_failed / cancelled items as pending again."""
        if self.job is None or self.is_running:
            return 0
        n = 0
        with self.job.lock:
            for it in self.job.items:
                if it.status in RETRYABLE:
                    it.status = ItemStatus.PENDING
                    it.error = None
                    n += 1
        return n

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)


def add_items(job: Job, pages: list[tuple[str, int, Any]]) -> list[Item]:
    """Save page images into the job's upload folder and create items."""
    from utils.file_utils import safe_stem
    job.upload_dir.mkdir(parents=True, exist_ok=True)
    multi = {}
    for name, page, _ in pages:
        multi[name] = multi.get(name, 0) + 1
    items = []
    for idx, (name, page, image) in enumerate(pages, 1):
        stem = safe_stem(name) + (f"_p{page}" if multi[name] > 1 else "")
        base, n = stem, 2
        while any(it.output_stem == stem for it in job.items + items):
            stem = f"{base}_{n}"
            n += 1
        path = job.upload_dir / f"{stem}.png"
        image.save(path)
        items.append(Item(item_id=f"{idx:04d}", source_name=name, page=page, image_path=str(path), output_stem=stem,
                          multipage=multi[name] > 1))
    job.items.extend(items)
    return items
