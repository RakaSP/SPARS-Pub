# Utils/Logger.py
from __future__ import annotations
import logging

# ----- define TRACE level -----
TRACE_LEVEL_NUM = 5
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")


def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, msg, args, **kwargs)


logging.Logger.trace = _trace  # add .trace() method to Logger

# ----- module-level logger -----
_logger = logging.getLogger("spars")

if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    _logger.addHandler(_handler)
    _logger.propagate = False  # don’t double-log to root

_LEVELS = {
    "TRACE": TRACE_LEVEL_NUM,
    "INFO": logging.INFO,
}


def set_log_level(level: str) -> None:
    """Set global log level: 'TRACE' or 'INFO'."""
    lvl = str(level).strip().upper()
    if lvl not in _LEVELS:
        raise ValueError("Unknown log level. Use TRACE or INFO.")
    level_num = _LEVELS[lvl]
    _logger.setLevel(level_num)
    for h in _logger.handlers:
        h.setLevel(level_num)


def log_trace(msg: str, *args, **kwargs) -> None:
    _logger.trace(msg, *args, **kwargs)


def log_info(msg: str, *args, **kwargs) -> None:
    _logger.info(msg, *args, **kwargs)


def get_logger() -> logging.Logger:
    """Return the underlying logger if you need direct access."""
    return _logger
