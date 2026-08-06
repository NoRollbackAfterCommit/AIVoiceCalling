"""Structured JSON logging.

Call logs are the audit trail for a government deployment, so every record
carries the call id when one is in scope. The id is threaded through a contextvar
rather than passed around, because it has to survive the async hand-offs between
the STT, agent and TTS tasks.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

call_id_var: ContextVar[str | None] = ContextVar("call_id", default=None)

_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


class _SafeAdapter(logging.LoggerAdapter):
    """Stops a structured field from colliding with a LogRecord attribute.

    `logging.makeRecord` raises KeyError if `extra` contains a reserved name —
    `args`, `module`, `name`, `filename` and friends — all of which are
    completely natural things to want to log about a tool call. Worse, the raise
    only happens once the level is actually enabled, so the bug sits invisible
    through a test suite running at WARNING and detonates the first time someone
    turns on INFO in production. Colliding keys are prefixed instead.
    """

    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        extra = kwargs.get("extra")
        if extra:
            kwargs["extra"] = {
                (f"ctx_{k}" if k in _RESERVED else k): v for k, v in extra.items()
            }
        return msg, kwargs


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        call_id = call_id_var.get()
        if call_id:
            payload["call_id"] = call_id
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # uvicorn installs its own colourised handlers; take them over so the whole
    # process emits one parseable stream for the ELK stack.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers[:] = []
        lg.propagate = True


def get_logger(name: str) -> logging.LoggerAdapter:
    return _SafeAdapter(logging.getLogger(name), {})
