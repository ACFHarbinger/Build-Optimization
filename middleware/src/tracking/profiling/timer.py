"""
Named block and function-level timing utilities.

Provides lightweight wrappers for timing arbitrary code sections and
automatically forwarding elapsed times to the active WSTracker run.

Classes:
    BlockTimer: Context-manager timer for a single named code block.
    MultiStepTimer: Multi-phase timer that tracks several sequential phases.

Functions:
    profile_block: Convenience context-manager wrapper around BlockTimer.
    profile_function: Decorator that times every call to a function.
"""

from __future__ import annotations

import contextlib
import functools
import time
from typing import Any, Callable, Dict, Generator, List, Optional


class BlockTimer:
    """Times a named code block and optionally logs elapsed time to WSTracker.

    Usage as a context manager::

        with BlockTimer("data_loading") as t:
            dataset = load_data()
        print(t.elapsed)   # seconds

    Usage as a standalone object::

        t = BlockTimer("training_epoch")
        t.start()
        train_one_epoch()
        t.stop()           # logs automatically

    Args:
        name: Human-readable label for the timed block.
        log_metric: When ``True`` (default), call :meth:`log_to_run` upon
            :meth:`stop`.
        step: Metric step dimension forwarded to ``Run.log_metric``.
        prefix: Metric key prefix.  The value is logged as
            ``{prefix}/{name}_sec``.
    """

    def __init__(
        self,
        name: str,
        log_metric: bool = True,
        step: int = 0,
        prefix: str = "time",
    ) -> None:
        self.name = name
        self.log_metric = log_metric
        self.step = step
        self.prefix = prefix
        self._start: Optional[float] = None
        self.elapsed: float = 0.0

    def start(self) -> "BlockTimer":
        """Record start time and return *self* for chaining."""
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        """Record stop time, compute elapsed, and optionally log to run.

        Returns:
            Elapsed wall-clock seconds.
        """
        if self._start is not None:
            self.elapsed = time.perf_counter() - self._start
            self._start = None
        if self.log_metric:
            self.log_to_run()
        return self.elapsed

    def log_to_run(self) -> None:
        """Forward the elapsed time to the active WSTracker run (silent no-op
        when no run is active or an error occurs)."""
        with contextlib.suppress(Exception):
            from tracking.core.run import get_active_run

            run = get_active_run()
            if run is not None:
                run.log_metric(f"{self.prefix}/{self.name}_sec", self.elapsed, step=self.step)

    def __enter__(self) -> "BlockTimer":
        return self.start()

    def __exit__(self, *_args: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        return f"BlockTimer(name={self.name!r}, elapsed={self.elapsed:.4f}s)"


class MultiStepTimer:
    """Multi-phase timer that tracks and accumulates named sub-phases.

    Calling :meth:`start` with a new phase name implicitly stops the previous
    one.  Call :meth:`stop` when the final phase is finished.

    Usage::

        t = MultiStepTimer()
        t.start("load")
        load_data()
        t.start("preprocess")   # stops "load", starts "preprocess"
        preprocess()
        t.start("train")
        train()
        t.stop()

        print(t)                        # per-phase summary
        t.log_to_run(prefix="epoch")    # emit to WSTracker

    Args:
        accumulate: When ``True`` (default), multiple :meth:`start` calls for
            the same phase name add to the existing total rather than
            overwriting it.
    """

    def __init__(self, accumulate: bool = True) -> None:
        self.accumulate = accumulate
        self._phases: Dict[str, List[float]] = {}
        self._current_phase: Optional[str] = None
        self._phase_start: Optional[float] = None

    def start(self, phase: str) -> "MultiStepTimer":
        """Begin timing *phase*.  Implicitly stops the current phase."""
        if self._current_phase is not None:
            self._commit()
        self._current_phase = phase
        self._phase_start = time.perf_counter()
        return self

    def stop(self) -> "MultiStepTimer":
        """Stop the current phase."""
        if self._current_phase is not None:
            self._commit()
            self._current_phase = None
            self._phase_start = None
        return self

    def _commit(self) -> None:
        if self._phase_start is None or self._current_phase is None:
            return
        elapsed = time.perf_counter() - self._phase_start
        if self.accumulate:
            self._phases.setdefault(self._current_phase, []).append(elapsed)
        else:
            self._phases[self._current_phase] = [elapsed]
        self._phase_start = None

    @property
    def total(self) -> float:
        """Total accumulated time across all phases (seconds)."""
        return sum(sum(v) for v in self._phases.values())

    def phase_total(self, phase: str) -> float:
        """Accumulated time for *phase* (0.0 if not recorded)."""
        return sum(self._phases.get(phase, [0.0]))

    def summary(self) -> Dict[str, float]:
        """Return ``{phase: total_seconds}`` for every recorded phase."""
        return {phase: sum(times) for phase, times in self._phases.items()}

    def log_to_run(self, prefix: str = "time", step: int = 0) -> None:
        """Log per-phase totals and overall total to the active WSTracker run."""
        with contextlib.suppress(Exception):
            from tracking.core.run import get_active_run

            run = get_active_run()
            if run is not None:
                metrics: Dict[str, Any] = {f"{prefix}/{phase}_sec": total for phase, total in self.summary().items()}
                metrics[f"{prefix}/total_sec"] = self.total
                run.log_metrics(metrics, step=step)

    def __repr__(self) -> str:
        total = self.total
        lines = [f"MultiStepTimer total={total:.4f}s:"]
        for phase, t in self.summary().items():
            pct = 100.0 * t / total if total > 0 else 0.0
            lines.append(f"  {phase}: {t:.4f}s ({pct:.1f}%)")
        return "\n".join(lines)


@contextlib.contextmanager
def profile_block(
    name: str,
    log_metric: bool = True,
    step: int = 0,
    prefix: str = "time",
) -> Generator[BlockTimer, None, None]:
    """Context manager that times a named code block.

    Args:
        name: Human-readable label for this block.
        log_metric: When ``True`` (default), log elapsed time to the active
            WSTracker run on exit.
        step: Metric step dimension (e.g. epoch or day number).
        prefix: Metric key prefix; the value is stored as
            ``{prefix}/{name}_sec``.

    Yields:
        :class:`BlockTimer` — inspect ``.elapsed`` after the block exits.

    Example::

        with profile_block("policy_inference", step=epoch) as t:
            action = policy(obs)
        print(t.elapsed)
    """
    timer = BlockTimer(name, log_metric=False, step=step, prefix=prefix)
    timer.start()
    try:
        yield timer
    finally:
        timer.elapsed = time.perf_counter() - timer._start if timer._start is not None else 0.0
        timer._start = None
        if log_metric:
            timer.log_to_run()


def profile_function(
    name: Optional[str] = None,
    log_metric: bool = True,
    prefix: str = "time",
) -> Callable[..., Any]:
    """Decorator that times every call to a function and logs to WSTracker.

    Args:
        name: Override metric key name.  Defaults to ``fn.__name__``.
        log_metric: When ``True`` (default), log elapsed time per call.
        prefix: Metric key prefix; logged as ``{prefix}/{name}_sec``.

    Returns:
        A decorator.  Must always be called with parentheses::

            @profile_function()
            def run_policy(obs):
                return policy(obs)

            @profile_function(prefix="sim", name="policy_step")
            def run_policy(obs):
                return policy(obs)
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        metric_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with profile_block(metric_name, log_metric=log_metric, prefix=prefix):
                return fn(*args, **kwargs)

        return wrapper

    return decorator
