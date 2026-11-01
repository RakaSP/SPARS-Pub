from __future__ import annotations

import logging
from typing import Any


TRACE_LEVEL_NUM: int = 5
_TRACE_LEVEL_NAME = "TRACE"
_QUIET_LEVEL_NAME = "QUIET"

logging.addLevelName(TRACE_LEVEL_NUM, _TRACE_LEVEL_NAME)


def _trace(
    self: logging.Logger,
    msg: object,
    *args: object,
    **kwargs: Any,
) -> None:
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, msg, args, **kwargs)


# Expose TRACE on Logger instances while keeping the standard logging backend.
logging.Logger.trace = _trace

_logger = logging.getLogger("spars")


_ALLOWED_LEVELS: dict[str, frozenset[int]] = {
    "TRACE": frozenset({TRACE_LEVEL_NUM}),
    "INFO": frozenset({TRACE_LEVEL_NUM, logging.INFO}),
    "WARNING": frozenset({TRACE_LEVEL_NUM, logging.INFO, logging.WARNING}),
    "ERROR": frozenset({
        TRACE_LEVEL_NUM,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
    }),
    _QUIET_LEVEL_NAME: frozenset({logging.ERROR}),
}


class _ExactLevelFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.allowed_levels = _ALLOWED_LEVELS["INFO"]

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno in self.allowed_levels


_level_filter = _ExactLevelFilter()
_enabled_levels = _ALLOWED_LEVELS["INFO"]

if not _logger.handlers:
    _stream_handler = logging.StreamHandler()
    _stream_handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(message)s")
    )
    _stream_handler.addFilter(_level_filter)
    _logger.addHandler(_stream_handler)
    _logger.propagate = False

# Keep the logger open down to TRACE and let the handler filter implement the
# SPARS cumulative semantics (TRACE only, INFO adds INFO, etc.).
_logger.setLevel(TRACE_LEVEL_NUM)
for _handler in _logger.handlers:
    _handler.setLevel(TRACE_LEVEL_NUM)


def normalize_log_level(level: object) -> str:
    normalized_level = str(level).strip().upper()

    if normalized_level not in _ALLOWED_LEVELS:
        raise ValueError(
            "Unknown log level. Use TRACE, INFO, WARNING, ERROR, or QUIET."
        )

    return normalized_level


def is_quiet_level(level: object) -> bool:
    return str(level).strip().upper() == _QUIET_LEVEL_NAME


def configure_logging(level: object) -> None:
    global _enabled_levels
    normalized_level = normalize_log_level(level)
    _enabled_levels = _ALLOWED_LEVELS[normalized_level]
    _level_filter.allowed_levels = _enabled_levels


def set_log_level(level: object) -> None:
    configure_logging(level)


def log_trace(
    msg: str,
    *args: object,
    **kwargs: Any,
) -> None:
    if TRACE_LEVEL_NUM in _enabled_levels:
        _logger.trace(msg, *args, **kwargs)


def log_info(
    msg: str,
    *args: object,
    **kwargs: Any,
) -> None:
    if logging.INFO in _enabled_levels:
        _logger.info(msg, *args, **kwargs)


def log_error(
    msg: str,
    *args: object,
    **kwargs: Any,
) -> None:
    if logging.ERROR in _enabled_levels:
        _logger.error(msg, *args, **kwargs)



