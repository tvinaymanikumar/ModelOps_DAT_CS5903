"""Thread-safe rate limiter: requests-per-minute window plus minimum delay.
Every attempt (including retries) must call acquire()."""
from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, requests_per_minute: float = 0, min_delay_s: float = 0.0):
        self.rpm = float(requests_per_minute or 0)
        self.min_delay_s = float(min_delay_s or 0)
        self._times: deque[float] = deque()
        self._last = 0.0
        self._lock = threading.Lock()
        self.total_acquired = 0

    def _wait_needed(self, now: float) -> float:
        wait = 0.0
        if self.min_delay_s and self._last:
            wait = max(wait, self._last + self.min_delay_s - now)
        if self.rpm > 0:
            while self._times and now - self._times[0] >= 60.0:
                self._times.popleft()
            if len(self._times) >= self.rpm:
                wait = max(wait, 60.0 - (now - self._times[0]))
        return wait

    def acquire(self, cancel_event: threading.Event | None = None, sleep_step: float = 0.1) -> bool:
        """Block until a request may be sent. Returns False if cancelled while waiting."""
        while True:
            if cancel_event is not None and cancel_event.is_set():
                return False
            with self._lock:
                now = time.monotonic()
                wait = self._wait_needed(now)
                if wait <= 0:
                    self._times.append(now)
                    self._last = now
                    self.total_acquired += 1
                    return True
            time.sleep(min(sleep_step, wait))
