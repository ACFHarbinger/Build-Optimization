"""WSTracker — centralized MLFlow-style experiment tracking for WSmart-Route.

Public API
----------
* :func:`init` — initialise the global tracker and return it.
* :func:`get_tracker` — return the active :class:`Tracker`, or ``None``.
* :func:`get_active_run` — return the active :class:`Run` for this process.
* :func:`init_worker` — attach an existing run in a worker sub-process.
* :class:`Tracker` — manages experiments and creates runs.
* :class:`Run` — logs metrics, params, tags, artifacts, and dataset events.

Submodules
----------
* :mod:`tracking.logging` — logging utilities (canonical location).
* :mod:`tracking.profiling` — execution profiling utilities.
* :mod:`tracking.hooks` — PyTorch forward/backward hooks.

  Both ``tracking.logging``, ``tracking.profiling``, and
  ``tracking.hooks`` are backwards-compatible shims that re-export from
  these canonical locations.

Typical usage (training)
------------------------
::

    import tracking as wst

    tracker = wst.init(
        experiment_name="AM-VRPP-50",
        tracking_uri="assets/tracking",
        run_type="training",
        tags={"model": "am", "problem": "vrpp"},
    )
    with tracker.start_run("AM-VRPP-50", run_type="training") as run:
        run.log_params({"lr": 1e-4, "batch_size": 256})
        trainer.fit(model)          # TrackingCallback logs metrics automatically
        run.log_artifact(model_path, artifact_type="model")

Typical usage (simulation, worker process)
------------------------------------------
::

    # Worker process — called via init_single_sim_worker
    wst.init_worker(tracking_uri="assets/tracking", run_id="<uuid>")

    run = wst.get_active_run()
    if run:
        run.log_metric("gurobi/s0/profit", 123.4, step=day)

Typical usage (hooks integration)
----------------------------------
::

    from tracking.hooks import add_gradient_monitoring_hooks, register_hooks_with_run
    hook_data = add_gradient_monitoring_hooks(model)
    loss.backward()
    run = wst.get_active_run()
    register_hooks_with_run(hook_data, run, prefix="train/hooks")

Typical usage (profiling)
--------------------------
::

    from tracking.profiling import start_global_profiling, stop_global_profiling
    start_global_profiling()
    train(...)
    stop_global_profiling()  # logs CSV as artifact to active run automatically
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from . import logging, profiling
from .core.run import Run, get_active_run, set_active_run
from .core.tracker import Tracker, get_tracker
from .integrations.filesystem import FilesystemTracker
from .integrations.mlflow_bridge import MLflowBridge
from .integrations.simulation import SimulationRunTracker, get_sim_tracker
from .viz_mixin import PolicyStateRecorder, PolicyVizMixin

__all__ = [
    # Factory / accessors
    "init",
    "init_worker",
    "get_tracker",
    "get_active_run",
    # Core classes
    "Tracker",
    "Run",
    # Integrations
    "TrackingCallback",
    "SimulationRunTracker",
    "get_sim_tracker",
    "RuntimeDataTracker",
    "FilesystemTracker",
    "MLflowBridge",
    "ZenMLBridge",
    # Submodules
    "hooks",
    "logging",
    "profiling",
    # Policy telemetry
    "PolicyVizMixin",
    "PolicyStateRecorder",
]

# ---------------------------------------------------------------------------
# Lazy imports (PEP 562) for PyTorch-dependent members.
#
# `tracking.core`/`profiling`/`logging`/most `integrations` are usable for
# lean, non-ML workloads (e.g. Build-Optimization's solver pipeline) without
# installing torch/pytorch-lightning/lightning/zenml. `hooks`,
# `integrations.data`, `integrations.lightning`, and `integrations.zenml_bridge`
# pull in that stack transitively and are deferred so importing `tracking`
# itself never requires any of it to be installed.
# ---------------------------------------------------------------------------

_LAZY_MODULES = {"hooks": "tracking.hooks"}
_LAZY_ATTRS = {
    "RuntimeDataTracker": ("tracking.integrations.data", "RuntimeDataTracker"),
    "TrackingCallback": ("tracking.integrations.lightning", "TrackingCallback"),
    "ZenMLBridge": ("tracking.integrations.zenml_bridge", "ZenMLBridge"),
}


def __getattr__(name: str) -> Any:
    import importlib

    if name in _LAZY_MODULES:
        module = importlib.import_module(_LAZY_MODULES[name])
        globals()[name] = module  # cache on the module so repeated access is direct
        return module

    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value  # cache on the module so repeated access is direct
    return value

# ---------------------------------------------------------------------------
# Module-level initialisation helpers
# ---------------------------------------------------------------------------

_DEFAULT_TRACKING_URI = "assets/tracking"


def init(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    run_type: str = "generic",
    tags: Optional[Dict[str, str]] = None,
    description: str = "",
    buffer_size: int = 200,
) -> Tracker:
    """Initialise the global :class:`Tracker` and return it.

    This is the primary entry-point for the main process.  Calling it a
    second time replaces the module-level tracker singleton.

    Args:
        experiment_name: Name of the experiment to create or resume.
        tracking_uri: Directory for ``tracking.db`` and artifact storage.
            Defaults to ``assets/tracking`` relative to ``ROOT_DIR``.
        run_type: Semantic run type (``'training'``, ``'simulation'``, …).
        tags: Tags to attach to every new run (informational only here;
            tags are applied per-run via :meth:`Tracker.start_run`).
        description: Human-readable experiment description.
        buffer_size: Metric buffer size per run before a DB flush.

    Returns:
        The initialised :class:`Tracker` singleton.
    """
    import tracking.core.tracker as _tracker_mod

    if tracking_uri is None:
        try:
            import os
            from pathlib import Path

            from constants import ROOT_DIR

            base_uri = "test_tracking" if os.environ.get("TEST_MODE") == "true" else _DEFAULT_TRACKING_URI
            tracking_uri = str(Path(ROOT_DIR).joinpath(base_uri).resolve())
        except ImportError:
            tracking_uri = _DEFAULT_TRACKING_URI

    tracker = Tracker(tracking_uri=tracking_uri, buffer_size=buffer_size)
    _tracker_mod._tracker = tracker
    return tracker


def init_worker(
    tracking_uri: str,
    run_id: str,
    buffer_size: int = 200,
) -> Optional[Run]:
    """Attach an existing run in a **worker sub-process**.

    Called inside ``init_single_sim_worker`` to give parallel simulation
    workers write access to the parent run without creating a new record.

    Args:
        tracking_uri: Same ``tracking_uri`` used by the parent process.
        run_id: UUID of the run created by the parent process.
        buffer_size: Metric buffer size for this worker.

    Returns:
        The attached :class:`Run` (now also set as the process-local active
        run), or ``None`` if *run_id* is not found.
    """
    import tracking.core.tracker as _tracker_mod

    try:
        tracker = Tracker(tracking_uri=tracking_uri, buffer_size=buffer_size)
        _tracker_mod._tracker = tracker
        run = tracker.attach_run(run_id)
        return run
    except Exception:  # noqa: BLE001
        return None
