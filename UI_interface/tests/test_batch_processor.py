import json
import threading
import time

from PIL import Image

from backend.job import ItemStatus, Job
from backend.processing_manager import ProcessingManager, add_items
from backend.rate_limiter import RateLimiter
from tests.helpers import make_task, prompt_for
from models.mock_adapter import MockAdapter


def _job(tmp_path, monkeypatch, n=3, mode="schema_based"):
    import backend.job as jobmod
    monkeypatch.setattr(jobmod, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(jobmod, "UPLOADS_DIR", tmp_path / "uploads")
    job = Job(task=make_task(mode), model_name="Mock Model", prompt=prompt_for(mode), params={},
              output_formats=["json", "txt", "pdf"])
    pages = [(f"doc{i}.png", 1, Image.new("RGB", (80 + i, 60), "white")) for i in range(n)]
    add_items(job, pages)
    return job


def test_sequential_run_saves_each_item(tmp_path, monkeypatch):
    job = _job(tmp_path, monkeypatch)
    mgr = ProcessingManager()
    mgr.set_job(job)
    assert mgr.start(adapter=MockAdapter(delay_s=0.05))
    assert not mgr.start(adapter=MockAdapter(delay_s=0.05))  # no duplicate worker
    mgr.join(10)
    assert job.finished
    assert all(it.status in (ItemStatus.COMPLETED, ItemStatus.NEEDS_REVIEW) for it in job.items)
    for it in job.items:
        env = json.loads((job.output_dir / "json" / f"{it.output_stem}.json").read_text())
        assert list(env["result"]) == list(job.task.template)
        assert env["mode"] == "schema_based" and env["task_name"] == "Test task"
        assert (job.output_dir / "pdf" / f"{it.output_stem}.pdf").exists()
        assert (job.output_dir / "raw" / f"{it.output_stem}.txt").exists()
    assert (job.output_dir / "summary" / "summary.csv").exists()


def test_stop_cancels_remaining(tmp_path, monkeypatch):
    job = _job(tmp_path, monkeypatch, n=6)
    mgr = ProcessingManager()
    mgr.set_job(job)
    mgr.start(adapter=MockAdapter(delay_s=0.4))
    time.sleep(0.5)
    mgr.stop()
    mgr.join(10)
    statuses = [it.status for it in job.items]
    assert ItemStatus.CANCELLED in statuses
    assert statuses.count(ItemStatus.CANCELLED) >= 3


def test_failures_then_retry(tmp_path, monkeypatch):
    job = _job(tmp_path, monkeypatch, n=2)
    mgr = ProcessingManager()
    mgr.set_job(job)
    mgr.start(max_retries=0, adapter=MockAdapter(delay_s=0, failure_rate=1.0))
    mgr.join(10)
    assert all(it.status == ItemStatus.FAILED for it in job.items)
    assert mgr.reset_retryable() == 2
    mgr.start(adapter=MockAdapter(delay_s=0))
    mgr.join(10)
    assert all(it.status in (ItemStatus.COMPLETED, ItemStatus.NEEDS_REVIEW) for it in job.items)


def test_rate_limiter_counts_every_acquire():
    rl = RateLimiter(requests_per_minute=2)
    assert rl.acquire() and rl.acquire()
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    assert rl.acquire(cancel) is False  # third request must wait; cancelled while waiting
    assert rl.total_acquired == 2


def test_min_delay():
    rl = RateLimiter(min_delay_s=0.3)
    t = time.monotonic()
    rl.acquire(); rl.acquire()
    assert time.monotonic() - t >= 0.28
