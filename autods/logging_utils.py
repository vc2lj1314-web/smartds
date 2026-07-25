"""Logging setup.

The original script shadowed the built-in ``print`` so that every message went
to both stdout and a log file. That made it impossible to use ``print`` for
anything else and hid the call site of each message. Here every module obtains a
named logger instead, and ``setup_logging`` wires those loggers to a file and to
stdout once, at process start.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(log_path: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger with a stdout handler and an optional file handler.

    Existing handlers are removed first so that repeated calls (for example in a
    notebook) do not duplicate every line.
    """
    root = logging.getLogger()
    root.setLevel(level)

    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_LOG_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_path:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    """Return the module-level logger for ``name``."""
    return logging.getLogger(name)
