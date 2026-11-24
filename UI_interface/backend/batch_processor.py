"""Sequential batch processing: one image per model request.

Runs in a background thread. Stop scheduling new images when cancelled; an
in-flight request is allowed to finish. Each completed item is saved
immediately.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from PIL import Image

from backend.job import Item, ItemStatus, Job
from backend.output_manager import OutputManager
from backend.rate_limiter import RateLimiter
from models.base_adapter import AdapterError, BaseAdapter, CancelledError, NotConfiguredError, TransientAdapterError
from processors.base_processor import BaseProcessor, ProcessingContext, ProcessOutput
from utils.image_utils import preprocess
from utils.logger import get_logger

log = get_logger(__name__)


class BatchProcessor:
    def __init__(self, job: Job, processor: BaseProcessor, adapter: BaseAdapter, rate_limiter: RateLimiter,
                 cancel_event: threading.Event, output_manager: OutputManager | None = None,
                 max_retries: int = 2, backoff_s: float = 2.0,
                 on_update: Callable[[Job], None] | None = None):
        self.job = job
        self.processor = processor
        self.adapter = adapter
        self.limiter = rate_limiter
        self.cancel = cancel_event
        self.out = output_manager or OutputManager(job)
        self.max_retries = int(max_retries)
        self.backoff_s = float(backoff_s)
        self.on_update = on_update or (lambda j: None)

    def _set(self, item: Item, **changes) -> None:
        with self.job.lock:
            for k, v in changes.items():
                setattr(item, k, v)
        self.on_update(self.job)

    def run(self) -> None:
        log.info("job %s started: mode=%s model=%s items=%d", self.job.job_id, self.job.task.mode,
                 self.job.model_name, len(self.job.items))
        self.out.write_job_settings()
        try:
            for item in list(self.job.items):
                if item.status != ItemStatus.PENDING:
                    continue
                if self.cancel.is_set():
                    break
                self._process_item(item)
        finally:
            with self.job.lock:
                for it in self.job.items:
                    if it.status in (ItemStatus.PENDING, ItemStatus.RUNNING):
                        it.status = ItemStatus.CANCELLED
                self.job.current_item_id = None
                self.job.finished = True
            self.out.write_summary()
            self.on_update(self.job)
            log.info("job %s finished: %s", self.job.job_id, self.job.counts())

    def _process_item(self, item: Item) -> None:
        with self.job.lock:
            self.job.current_item_id = item.item_id
            item.status = ItemStatus.RUNNING
            item.error = None
        self.on_update(self.job)
        try:
            image = Image.open(item.image_path)
            image.load()
            image = preprocess(image, self.job.params.get("preprocess", {}))
        except Exception as exc:
            self._finish_failed(item, f"Could not read image: {type(exc).__name__}")
            return

        ctx = ProcessingContext(task=self.job.task, adapter=self.adapter, model_name=self.job.model_name,
                                prompt=self.job.prompt, params=self.job.params, cancel_event=self.cancel)
        attempt = 0
        while True:
            attempt += 1
            if not self.limiter.acquire(self.cancel):
                self._set(item, status=ItemStatus.CANCELLED, attempts=attempt - 1)
                return
            self._set(item, attempts=attempt)
            try:
                output = self.processor.process(image, ctx)
                break
            except CancelledError:
                self._set(item, status=ItemStatus.CANCELLED)
                return
            except TransientAdapterError as exc:
                if attempt > self.max_retries or self.cancel.is_set():
                    self._finish_failed(item, f"{exc} (after {attempt} attempts)")
                    return
                wait = exc.retry_after or self.backoff_s * (2 ** (attempt - 1))
                log.warning("item %s transient error, retry in %.1fs: %s", item.item_id, wait, exc)
                end = time.monotonic() + wait
                while time.monotonic() < end and not self.cancel.is_set():
                    time.sleep(0.1)
            except NotConfiguredError as exc:
                self._finish_failed(item, str(exc))
                return
            except AdapterError as exc:
                self._finish_failed(item, str(exc))
                return
            except Exception as exc:  # never let one item kill the worker
                log.exception("item %s unexpected error", item.item_id)
                self._finish_failed(item, f"{type(exc).__name__}: {exc}")
                return
        self._finish(item, output)

    def _finish(self, item: Item, output: ProcessOutput) -> None:
        status = ItemStatus(output.status)
        envelope = self.out.save_item(item, output)
        self._set(item, status=status, error=output.error, latency_ms=output.latency_ms,
                  flags=output.flags, envelope=envelope)

    def _finish_failed(self, item: Item, message: str) -> None:
        output = ProcessOutput(status="failed", result=None, raw_text="", error=message)
        envelope = self.out.save_item(item, output)
        self._set(item, status=ItemStatus.FAILED, error=message, envelope=envelope)
