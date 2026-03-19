"""
Configurable logging utility.

Provides a ``get_logger`` factory that returns a ``logging.Logger`` with
console and (optional) file handlers.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


_INITIALISED_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(
    name: str = "sensor_fusion",
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Return a named logger with console and optional file output.

    Parameters
    ----------
    name : str
        Logger name.
    level : str
        Logging level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
    log_file : str or None
        If provided, logs are also written to this file.

    Returns
    -------
    logging.Logger
    """
    if name in _INITIALISED_LOGGERS:
        return _INITIALISED_LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _INITIALISED_LOGGERS[name] = logger
    return logger
