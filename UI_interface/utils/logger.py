"""Logger that redacts secrets. Never log image bytes, extracted values or
raw responses; log job IDs, file names, model, status, latency and error types."""
from __future__ import annotations

import logging
import os
import re

_SECRET_ENV = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
_PATTERNS = [re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+"), re.compile(r"(?i)(api[_-]?key=)[^&\s]+"),
             re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), re.compile(r"sk-[0-9A-Za-z]{16,}")]


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for pat in _PATTERNS:
            msg = pat.sub(lambda m: (m.group(1) if m.groups() else "") + "***", msg)
        for k, v in os.environ.items():
            if v and len(v) >= 8 and any(s in k.upper() for s in _SECRET_ENV):
                msg = msg.replace(v, "***")
        record.msg, record.args = msg, None
        return True


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler.addFilter(RedactingFilter())
        logger.addHandler(handler)
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger
