"""
Logging systems for terminal, file, and GUI communication.
This module acts as a facade, delegating to specialized sub-modules.
"""

from .modules.analysis import (
    final_simulation_summary,
    load_log_dict,
    output_stats,
    runs_per_policy,
)
from .modules.gui import send_daily_output_to_gui, send_final_output_to_gui
from .modules.metrics import get_loss_stats, log_epoch, log_values
from .modules.storage import (
    log_to_json,
    log_to_json2,
    log_to_pickle,
    setup_system_logger,
    sort_log,
    update_log,
)

__all__ = [
    "log_values",
    "log_epoch",
    "get_loss_stats",
    "setup_system_logger",
    "sort_log",
    "log_to_json",
    "log_to_json2",
    "log_to_pickle",
    "update_log",
    "load_log_dict",
    "output_stats",
    "runs_per_policy",
    "final_simulation_summary",
    "send_daily_output_to_gui",
    "send_final_output_to_gui",
]
