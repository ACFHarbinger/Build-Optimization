"""
Profiling utilities package.

Provides lightweight, WSTracker-integrated tools for measuring execution time,
memory usage, and throughput at multiple granularities:

* :class:`ExecutionProfiler` / :func:`start_global_profiling` /
  :func:`stop_global_profiling` — ``sys.setprofile`` hook that records every
  project function call to a buffered CSV.
* :class:`BlockTimer` / :func:`profile_block` — context-manager timer for
  named code sections.
* :class:`MultiStepTimer` — multi-phase timer that tracks sequential stages.
* :func:`profile_function` — decorator that times every call to a function.
* :class:`MemorySnapshot` — point-in-time GPU/CPU memory snapshot.
* :class:`MemoryTracker` — background-thread memory monitor.
* :class:`ThroughputTracker` — sliding-window items-per-second meter.
* :class:`ProfilingReport` — reads and analyses an :class:`ExecutionProfiler`
  CSV to produce top-function and per-module breakdowns.

All classes auto-forward results to the active WSTracker run via
``contextlib.suppress``-guarded calls — they are safe no-ops when no run is
active or optional dependencies are missing.

Example usage::

    from tracking.profiling import (
        profile_block,
        profile_function,
        MemorySnapshot,
        MemoryTracker,
        ThroughputTracker,
        start_global_profiling,
        stop_global_profiling,
    )

    # Time a single block
    with profile_block("data_loading", step=epoch):
        dataset = load_data()

    # Decorate a function
    @profile_function(prefix="sim")
    def run_policy(obs):
        return policy(obs)

    # Track memory during training
    with MemoryTracker(interval_sec=0.5, tag="train") as mem:
        train_one_epoch()
    mem.log_summary_to_run(step=epoch)

    # Measure throughput
    tracker = ThroughputTracker(window=50, unit="samples")
    tracker.start()
    for batch in dataloader:
        with tracker.step(len(batch)):
            loss = model(batch)
    tracker.log_to_run(step=epoch, prefix="train")

    # Full execution profiling
    start_global_profiling()
    run_experiment()
    stop_global_profiling()   # auto-logs CSV as artifact
"""

from typing import Any

from .profiler import ExecutionProfiler, start_global_profiling, stop_global_profiling
from .report import ProfilingReport
from .throughput import ThroughputTracker
from .timer import BlockTimer, MultiStepTimer, profile_block, profile_function

# MemorySnapshot/MemoryTracker profile GPU tensor memory and require torch;
# deferred so the rest of `profiling` works without it installed.
_LAZY_ATTRS = {
    "MemorySnapshot": "MemorySnapshot",
    "MemoryTracker": "MemoryTracker",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module("tracking.profiling.memory")
    value = getattr(module, _LAZY_ATTRS[name])
    globals()[name] = value
    return value

__all__ = [
    # Execution profiler (sys.setprofile)
    "ExecutionProfiler",
    "start_global_profiling",
    "stop_global_profiling",
    # Block / function timers
    "BlockTimer",
    "MultiStepTimer",
    "profile_block",
    "profile_function",
    # Memory profiling
    "MemorySnapshot",
    "MemoryTracker",
    # Throughput tracking
    "ThroughputTracker",
    # Report generation
    "ProfilingReport",
]
