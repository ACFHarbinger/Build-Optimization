"""
Profiling report generation from ExecutionProfiler CSV output.

Reads the CSV produced by :class:`ExecutionProfiler`, computes per-function,
per-file, per-class, and per-module aggregates, detects timeline gaps, and can
forward a structured summary to the active WSTracker run.

Classes:
    ProfilingReport: Reads and analyses a profiling CSV.
"""

from __future__ import annotations

import contextlib
import csv
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


class ProfilingReport:
    """Reads and analyses a profiling CSV produced by :class:`ExecutionProfiler`.

    Provides:

    * :meth:`top_functions` — top-N hottest functions by total accumulated time.
    * :meth:`file_breakdown` — time grouped by source file.
    * :meth:`class_breakdown` — time grouped by class name.
    * :meth:`module_breakdown` — time grouped by top-level module directory.
    * :meth:`timeline_gaps` — periods where no function calls were recorded.
    * :meth:`log_to_run` — forward a compact summary to the active WSTracker run.

    Usage::

        report = ProfilingReport("logs/function_execution_times_2024-01-01_12-00-00.csv")
        print(report)
        for entry in report.top_functions(n=10):
            print(entry)
        report.log_to_run(top_n=10)

    Or get one directly from a running profiler::

        report = profiler.get_report()   # calls flush() first

    Args:
        csv_path: Path to the CSV file written by :class:`ExecutionProfiler`.
        wall_elapsed: Optional wall-clock seconds the profiler was active.
    """

    def __init__(
        self,
        csv_path: str,
        wall_elapsed: Optional[float] = None,
    ) -> None:
        self.csv_path = csv_path
        self.wall_elapsed = wall_elapsed
        self._records: List[Dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self.csv_path):
            return
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    self._records.append(
                        {
                            "timestamp": row.get("timestamp", ""),
                            "file": row.get("file", ""),
                            "class": row.get("class", ""),
                            "function": row.get("function", ""),
                            "duration_sec": float(row.get("duration_sec", 0)),
                        }
                    )
                except (ValueError, KeyError):
                    continue

    # ------------------------------------------------------------------
    # Basic statistics
    # ------------------------------------------------------------------

    @property
    def n_calls(self) -> int:
        """Total number of function-call records in the CSV."""
        return len(self._records)

    @property
    def total_time(self) -> float:
        """Sum of all recorded durations (seconds).

        Note: this is the sum of self-times *and* child-times, so the value
        will exceed the wall-clock elapsed time for recursive or nested code.
        Use :meth:`module_breakdown` or :meth:`top_functions` for relative
        comparisons.
        """
        return sum(r["duration_sec"] for r in self._records)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def top_functions(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return the top-*n* functions sorted by total accumulated time.

        Each entry is a dict with keys:

        * ``key`` — ``"{file}::{class}.{function}"`` (or without class)
        * ``total_sec`` — sum of all durations for this function
        * ``n_calls`` — number of times the function was called
        * ``avg_sec`` — mean duration per call
        * ``max_sec`` — longest single call

        Args:
            n: Maximum number of entries to return.
        """
        accum: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total_sec": 0.0, "n_calls": 0, "max_sec": 0.0})
        for r in self._records:
            key = f"{r['file']}::{r['class']}.{r['function']}" if r["class"] else f"{r['file']}::{r['function']}"
            entry = accum[key]
            entry["total_sec"] += r["duration_sec"]
            entry["n_calls"] += 1
            if r["duration_sec"] > entry["max_sec"]:
                entry["max_sec"] = r["duration_sec"]

        result: List[Dict[str, Any]] = []
        for key, stats in accum.items():
            result.append(
                {
                    "key": key,
                    "total_sec": stats["total_sec"],
                    "n_calls": stats["n_calls"],
                    "avg_sec": stats["total_sec"] / max(stats["n_calls"], 1),
                    "max_sec": stats["max_sec"],
                }
            )
        result.sort(key=lambda x: x["total_sec"], reverse=True)
        return result[:n]

    def file_breakdown(self, n: int = 15) -> List[Dict[str, Any]]:
        """Total time grouped by source file path.

        Returns a list of dicts with keys ``file``, ``total_sec``,
        ``n_calls``, sorted by total time descending.
        """
        accum: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total_sec": 0.0, "n_calls": 0})
        for r in self._records:
            entry = accum[r["file"]]
            entry["total_sec"] += r["duration_sec"]
            entry["n_calls"] += 1

        result = [{"file": f, "total_sec": s["total_sec"], "n_calls": s["n_calls"]} for f, s in accum.items()]
        result.sort(key=lambda x: x["total_sec"], reverse=True)
        return result[:n]

    def class_breakdown(self, n: int = 15) -> List[Dict[str, Any]]:
        """Total time grouped by class name.

        Returns a list of dicts with keys ``class``, ``total_sec``,
        ``n_calls``, sorted by total time descending.  Functions without
        a class are grouped under ``"<module-level>"``.
        """
        accum: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total_sec": 0.0, "n_calls": 0})
        for r in self._records:
            cls = r["class"] or "<module-level>"
            entry = accum[cls]
            entry["total_sec"] += r["duration_sec"]
            entry["n_calls"] += 1

        result = [{"class": c, "total_sec": s["total_sec"], "n_calls": s["n_calls"]} for c, s in accum.items()]
        result.sort(key=lambda x: x["total_sec"], reverse=True)
        return result[:n]

    def module_breakdown(self) -> List[Tuple[str, float]]:
        """Total time grouped by first path segment (top-level module directory).

        Returns a list of ``(module_name, total_seconds)`` tuples sorted by
        total time descending.
        """
        totals: Dict[str, float] = defaultdict(float)
        for r in self._records:
            parts = r["file"].replace("\\", "/").split("/")
            module = parts[0] if parts else "unknown"
            totals[module] += r["duration_sec"]
        return sorted(totals.items(), key=lambda x: x[1], reverse=True)

    def timeline_gaps(self, min_gap_sec: float = 1.0) -> List[Dict[str, Any]]:
        """Detect gaps in the profiling timeline larger than *min_gap_sec*.

        These gaps represent time spent in library code, I/O, child
        processes, or other code not instrumented by the profiler.

        Returns a list of dicts with keys ``after_index``, ``from_ts``,
        ``to_ts``, and ``gap_sec``, sorted by gap size descending.
        """
        if len(self._records) < 2:
            return []

        timestamps: List[Tuple[int, datetime]] = []
        for i, r in enumerate(self._records):
            ts_str = r["timestamp"]
            try:
                if "." in ts_str:
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                else:
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                timestamps.append((i, ts))
            except (ValueError, TypeError):
                continue

        if len(timestamps) < 2:
            return []

        gaps: List[Dict[str, Any]] = []
        for j in range(1, len(timestamps)):
            prev_idx, prev_ts = timestamps[j - 1]
            curr_idx, curr_ts = timestamps[j]
            delta = (curr_ts - prev_ts).total_seconds()
            if delta >= min_gap_sec:
                gaps.append(
                    {
                        "after_index": prev_idx,
                        "from_ts": prev_ts.strftime("%H:%M:%S.%f")[:12],
                        "to_ts": curr_ts.strftime("%H:%M:%S.%f")[:12],
                        "gap_sec": delta,
                    }
                )

        gaps.sort(key=lambda x: x["gap_sec"], reverse=True)
        return gaps

    def slowest_call(self) -> Optional[Dict[str, Any]]:
        """Return the single longest recorded call, or ``None`` if empty."""
        if not self._records:
            return None
        return max(self._records, key=lambda r: r["duration_sec"])

    def summary(self) -> Dict[str, Any]:
        """Return a high-level summary dict."""
        return {
            "n_calls": self.n_calls,
            "total_time_sec": self.total_time,
            "wall_elapsed_sec": self.wall_elapsed,
            "csv_path": self.csv_path,
        }

    # ------------------------------------------------------------------
    # WSTracker integration
    # ------------------------------------------------------------------

    def log_to_run(self, top_n: int = 10, step: int = 0) -> None:
        """Log a compact profiling summary to the active WSTracker run.

        Logged information:

        * **Params**: ``profiling.n_calls``, ``profiling.total_time_sec``,
          ``profiling.csv_path``, and for each top-*n* function:
          ``profiling.top{rank}.key``, ``profiling.top{rank}.total_sec``,
          ``profiling.top{rank}.n_calls``.
        * **Metrics**: ``profiling/module/{name}_sec`` for each module in
          :meth:`module_breakdown` (up to *top_n* entries).
        * **Artifact**: the CSV file itself is registered as type
          ``"profiling"``.

        Args:
            top_n: Number of top functions / modules to include.
            step: Metric step dimension.
        """
        with contextlib.suppress(Exception):
            from tracking.core.run import get_active_run

            run = get_active_run()
            if run is None:
                return

            params: Dict[str, Any] = {
                "profiling.n_calls": self.n_calls,
                "profiling.total_time_sec": round(self.total_time, 4),
                "profiling.csv_path": self.csv_path,
            }
            if self.wall_elapsed is not None:
                params["profiling.wall_elapsed_sec"] = round(self.wall_elapsed, 4)

            run.log_params(params)

            for rank, fn in enumerate(self.top_functions(top_n), start=1):
                run.log_params(
                    {
                        f"profiling.top{rank}.key": fn["key"][:120],
                        f"profiling.top{rank}.total_sec": round(fn["total_sec"], 6),
                        f"profiling.top{rank}.n_calls": fn["n_calls"],
                    }
                )

            for module, total in self.module_breakdown()[:top_n]:
                run.log_metric(f"profiling/module/{module}_sec", total, step=step)

            if os.path.exists(self.csv_path):
                run.log_artifact(self.csv_path, artifact_type="profiling")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"ProfilingReport(n_calls={self.n_calls}, total={self.total_time:.2f}s, path={self.csv_path!r})"

    def __str__(self) -> str:
        sep = "=" * 80
        lines = [
            sep,
            "EXECUTION PROFILING REPORT".center(80),
            sep,
            "",
        ]

        # --- Overview ---
        lines.append(f"  CSV:           {self.csv_path}")
        lines.append(f"  Total Calls:   {self.n_calls:,}")
        lines.append(f"  Sum of Durations:  {self.total_time:.3f}s")
        if self.wall_elapsed is not None:
            lines.append(f"  Wall Time:     {self.wall_elapsed:.3f}s")
            coverage = 100.0 * self.total_time / self.wall_elapsed if self.wall_elapsed > 0 else 0.0
            lines.append(f"  Coverage:      {coverage:.1f}% (profiled / wall)")
        lines.append("")

        # --- Top Functions ---
        lines.append("  Top 15 Functions by Accumulated Time:")
        lines.append(f"  {'#':>3}  {'Total':>9}  {'Calls':>7}  {'Avg':>9}  {'Max':>9}  Function")
        lines.append("  " + "-" * 76)
        for rank, fn in enumerate(self.top_functions(15), start=1):
            lines.append(
                f"  {rank:>3}  {fn['total_sec']:>8.3f}s  "
                f"{fn['n_calls']:>7,}  "
                f"{fn['avg_sec'] * 1000:>8.1f}ms "
                f"{fn['max_sec'] * 1000:>8.1f}ms "
                f" {fn['key']}"
            )
        lines.append("")

        # --- File Breakdown ---
        lines.append("  Top 10 Files by Accumulated Time:")
        lines.append(f"  {'Total':>9}  {'Calls':>7}  {'Avg':>9}  File")
        lines.append("  " + "-" * 76)
        for fb in self.file_breakdown(10):
            avg_ms = fb["total_sec"] / fb["n_calls"] * 1000 if fb["n_calls"] > 0 else 0.0
            lines.append(f"  {fb['total_sec']:>8.3f}s  {fb['n_calls']:>7,}  {avg_ms:>8.1f}ms  {fb['file']}")
        lines.append("")

        # --- Class Breakdown ---
        class_entries = self.class_breakdown(10)
        if class_entries:
            lines.append("  Top 10 Classes by Accumulated Time:")
            lines.append(f"  {'Total':>9}  {'Calls':>7}  Class")
            lines.append("  " + "-" * 50)
            for cb in class_entries:
                lines.append(f"  {cb['total_sec']:>8.3f}s  {cb['n_calls']:>7,}  {cb['class']}")
            lines.append("")

        # --- Module Breakdown ---
        lines.append("  Module Breakdown:")
        for module, total in self.module_breakdown():
            pct = 100.0 * total / self.total_time if self.total_time > 0 else 0.0
            lines.append(f"    {module:<30} {total:>8.3f}s  ({pct:.1f}%)")
        lines.append("")

        # --- Timeline Gaps ---
        gaps = self.timeline_gaps(min_gap_sec=1.0)
        if gaps:
            lines.append(f"  Timeline Gaps (>1s, {len(gaps)} found):")
            lines.append(f"  {'Gap':>8}  {'From':>12}  {'To':>12}  Description")
            lines.append("  " + "-" * 60)
            for g in gaps[:15]:
                lines.append(
                    f"  {g['gap_sec']:>7.1f}s  {g['from_ts']:>12}  {g['to_ts']:>12}  after record #{g['after_index']}"
                )
            if len(gaps) > 15:
                lines.append(f"  ... and {len(gaps) - 15} more gaps")
            lines.append("")

        lines.append(sep)
        return "\n".join(lines)
