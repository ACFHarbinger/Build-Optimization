"""Runtime data mutation tracking for WSTracker.

:class:`RuntimeDataTracker` attaches to a training run and records statistical
snapshots of tensor datasets each time they are loaded into memory, regenerated,
or mutated.  It captures the distribution of each tensor field (mean, std, min,
max, shape) and logs those as run params so drift across epochs is visible in
the experiment database.

This module is concerned exclusively with **in-memory variable state** — it
watches the Python objects that hold the data, not files on disk.  For
filesystem-level change detection (SHA-256 hashing, file events) see
:mod:`tracking.integrations.filesystem`.

Typical usage
-------------
::

    from tracking.integrations.data import RuntimeDataTracker
    import tracking as wst

    tracker = RuntimeDataTracker(wst.get_active_run())

    # Baseline snapshot right after data is loaded into memory
    tracker.on_load(dataset.data)

    # After each epoch regeneration, snapshot the new distribution
    new_dataset = regenerate_dataset(env, size)
    tracker.on_regenerate(new_dataset.data, epoch=1)

    # Inspect drift between the last two snapshots
    drift = tracker.field_drift("waste", stat="mean")
"""

from __future__ import annotations

import contextlib
import itertools
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import torch


class RuntimeDataTracker:
    """Tracks in-memory dataset mutations during a training or simulation run.

    Records per-field tensor statistics (mean, std, min, max, shape) as run
    params on every snapshot, making distribution shift across epochs visible
    in the tracking database.

    All param keys are namespaced as ``data/{tag}/{field}/{stat}`` so multiple
    datasets can be tracked in the same run without key collisions.

    Args:
        run: The active tracking run.  If ``None``, all methods are no-ops so
            the class can be used unconditionally without guard clauses.
        max_fields: Maximum number of tensor fields to snapshot per call.
            Protects against very wide batches flooding the param table.
    """

    def __init__(self, run: Any, max_fields: int = 32) -> None:
        self._run = run
        self._max_fields = max_fields
        self._history: List[Tuple[str, Dict[str, Any]]] = []

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def snapshot(
        self,
        data: Any,
        tag: str,
        step: Optional[int] = None,
    ) -> None:
        """Capture tensor statistics from *data* and log them to the run.

        *data* can be a :class:`torch.Tensor`, a plain ``dict`` mapping field
        names to tensors, or any object with a ``data`` attribute that is one
        of the above (e.g. a ``TensorDictDataset``).

        Args:
            data: The in-memory dataset payload to inspect.
            tag: Namespace prefix for param keys (e.g. ``"train/epoch_3"``).
            step: Optional integer step to also log per-field mean as a metric
                for trending purposes.
        """
        if self._run is None:
            return
        fields = self._extract_fields(data)
        stats = self._compute_stats(fields)
        self._log_stats(stats, tag, step)
        self._history.append((tag, stats))

    def on_load(
        self,
        data: Any,
        shape: Optional[tuple] = None,
        metadata: Optional[Dict[str, Any]] = None,
        log_event: bool = True,
    ) -> None:
        """Snapshot statistics of a dataset variable right after it is loaded.

        Call this immediately after the data object is constructed/loaded so
        the initial distribution is captured as the baseline.

        Args:
            data: The in-memory dataset variable (tensor, dict, or TensorDict).
            shape: Explicit shape; inferred from *data* if omitted.
            metadata: Extra context logged as a mutation record (e.g. problem,
                graph_size).
            log_event: When ``False``, skip the internal ``log_dataset_event``
                call.  Use this when the caller already logs its own event to
                avoid duplicates.
        """
        if self._run is None:
            return
        n = shape or self._infer_shape(data)
        if log_event:
            meta: Dict[str, Any] = {
                "event": "load",
                "shape": n,
            }
            if metadata:
                meta.update(metadata)
            self._run.log_dataset_event("load", shape=n, metadata=meta)
        self.snapshot(data, tag="load")

    def on_regenerate(
        self,
        data: Any,
        epoch: int,
        shape: Optional[tuple] = None,
        metadata: Optional[Dict[str, Any]] = None,
        log_event: bool = True,
    ) -> None:
        """Record that the training dataset variable was replaced for a new epoch.

        Logs a mutation event and snapshots the new variable's distribution so
        any shift from the previous epoch is visible in the params table.

        Args:
            data: The freshly generated in-memory dataset.
            epoch: Current training epoch (used in the snapshot tag and step).
            shape: Explicit shape; inferred from *data* if omitted.
            metadata: Extra context forwarded to the mutation record.
            log_event: When ``False``, skip the internal ``log_dataset_event``
                call to avoid duplicates.
        """
        if self._run is None:
            return
        n = shape or self._infer_shape(data)
        if log_event:
            meta: Dict[str, Any] = {
                "event": "regenerate",
                "epoch": epoch,
                "shape": n,
            }
            if metadata:
                meta.update(metadata)
            self._run.log_dataset_event("mutate", shape=n, metadata=meta)
        self.snapshot(data, tag=f"train/epoch_{epoch}", step=epoch)

    def on_augment(
        self,
        data: Any,
        description: str,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        log_event: bool = True,
    ) -> None:
        """Record an in-memory augmentation or transform applied to the dataset variable.

        Args:
            data: The dataset after the transform has been applied.
            description: Short human-readable label for the transform.
            step: Optional step value for metric logging.
            metadata: Extra context forwarded to the mutation record.
            log_event: When ``False``, skip the internal ``log_dataset_event``
                call to avoid duplicates.
        """
        if self._run is None:
            return
        if log_event:
            meta: Dict[str, Any] = {
                "event": "augment",
                "description": description,
            }
            if metadata:
                meta.update(metadata)
            self._run.log_dataset_event("mutate", metadata=meta)
        self.snapshot(data, tag=f"augment/{description}", step=step)

    # ------------------------------------------------------------------
    # Delta / drift helpers
    # ------------------------------------------------------------------

    def field_drift(
        self,
        field: str,
        stat: str = "mean",
        window: int = 2,
    ) -> Optional[float]:
        """Return the absolute change in *stat* for *field* over the last *window* snapshots.

        Useful for detecting distribution shift between dataset regenerations.

        Args:
            field: Tensor field name (e.g. ``"waste"``).
            stat: One of ``"mean"``, ``"std"``, ``"min"``, ``"max"``.
            window: Number of recent snapshots to compare (default 2 = last vs. previous).

        Returns:
            Absolute delta, or ``None`` if there are not enough snapshots or
            the field does not exist.
        """
        relevant = [s for _, s in self._history if field in s and stat in s[field]]
        if len(relevant) < window:
            return None
        prev = relevant[-window][field][stat]
        curr = relevant[-1][field][stat]
        with contextlib.suppress(TypeError, ValueError):
            return abs(float(curr) - float(prev))
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_fields(self, data: Any) -> Dict[str, torch.Tensor]:
        """Unwrap *data* into a flat ``{field: tensor}`` mapping."""
        if hasattr(data, "data"):
            data = data.data

        if isinstance(data, torch.Tensor):
            return {"data": data}

        if isinstance(data, dict):
            return {k: v for k, v in list(data.items())[: self._max_fields] if isinstance(v, torch.Tensor)}

        # TensorDict-like (tensordict library or similar mapping)
        if hasattr(data, "keys") and callable(data.keys):
            result: Dict[str, torch.Tensor] = {}
            for k in itertools.islice(cast(Iterable[Any], data.keys()), self._max_fields):
                with contextlib.suppress(Exception):
                    v = data[k]
                    if isinstance(v, torch.Tensor):
                        result[k] = v
            return result

        return {}

    @staticmethod
    def _compute_stats(fields: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
        """Compute per-field descriptive statistics."""
        stats: Dict[str, Dict[str, Any]] = {}
        for name, tensor in fields.items():
            t = tensor.float().detach()
            shape: List[int] = [int(d) for d in t.shape]
            try:
                entry: Dict[str, Any] = {
                    "shape": shape,
                    "numel": t.numel(),
                    "mean": round(t.mean().item(), 6),
                    "std": round(t.std().item(), 6) if t.numel() > 1 else 0.0,
                    "min": round(t.min().item(), 6),
                    "max": round(t.max().item(), 6),
                }
            except Exception:
                entry = {"shape": shape, "numel": t.numel()}
            stats[name] = entry
        return stats

    def _log_stats(
        self,
        stats: Dict[str, Dict[str, Any]],
        tag: str,
        step: Optional[int],
    ) -> None:
        """Flatten stats into run params (and optionally metrics)."""
        params: Dict[str, Any] = {}
        for field, entry in stats.items():
            for stat_name, val in entry.items():
                params[f"data/{tag}/{field}/{stat_name}"] = val

        if params:
            self._run.log_params(params)

        if step is not None:
            for field, entry in stats.items():
                if "mean" in entry:
                    with contextlib.suppress(Exception):
                        self._run.log_metric(f"data/{field}/mean", float(entry["mean"]), step=step)

    @staticmethod
    def _infer_shape(data: Any) -> Optional[tuple]:
        """Best-effort shape inference from a variable."""
        if hasattr(data, "shape"):
            with contextlib.suppress(Exception):
                return tuple(data.shape)
        if hasattr(data, "data") and hasattr(data.data, "shape"):
            with contextlib.suppress(Exception):
                return tuple(data.data.shape)
        with contextlib.suppress(Exception):
            return (int(len(data)),)
        return None
