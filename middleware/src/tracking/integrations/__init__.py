"""Pipeline integration helpers for WSTracker."""

from typing import Any

from .filesystem import FilesystemTracker
from .simulation import SimulationRunTracker, get_sim_tracker

__all__ = [
    "TrackingCallback",
    "SimulationRunTracker",
    "get_sim_tracker",
    "RuntimeDataTracker",
    "FilesystemTracker",
    "ZenMLBridge",
]

# RuntimeDataTracker/TrackingCallback are PyTorch-only, and ZenMLBridge pulls
# in `lightning.fabric` transitively via tracking.logging.pylogger; all three
# are deferred so the rest of `integrations` works without torch/lightning
# installed.
_LAZY_ATTRS = {
    "RuntimeDataTracker": ("tracking.integrations.data", "RuntimeDataTracker"),
    "TrackingCallback": ("tracking.integrations.lightning", "TrackingCallback"),
    "ZenMLBridge": ("tracking.integrations.zenml_bridge", "ZenMLBridge"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
