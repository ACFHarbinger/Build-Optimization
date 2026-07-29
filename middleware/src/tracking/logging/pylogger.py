"""
Multi-GPU friendly python logger.
Adapter from rl4co/utils/pylogger.py.
"""

import logging
from typing import Optional

from lightning.fabric.utilities.rank_zero import rank_zero_only

from tracking.logging.structured_logging import get_structured_logger as _get_structured


def get_pylogger(name=__name__, log_file: Optional[str] = None) -> logging.Logger:
    """
    Initializes multi-GPU-friendly python command line logger.

    Args:
        name: Logger name.
        log_file: Optional path to log file. If provided, enables structured JSON logging.
    """
    # If log_file is provided, use structured logger setup first
    logger = _get_structured(name=name, log_file=log_file) if log_file else logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
