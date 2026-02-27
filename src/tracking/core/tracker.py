"""Central experiment tracker — the main entry-point of WSTracker."""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

from .run import Run, set_active_run
from .store import TrackingStore

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_tracker: Optional["Tracker"] = None


def get_tracker() -> Optional["Tracker"]:
    """Return the process-global :class:`Tracker` instance, or ``None``."""
    return _tracker


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class Tracker:
    """Centralized MLFlow-style experiment tracker for WSmart-Route.

    A single :class:`Tracker` instance manages the SQLite database and
    coordinates the creation of :class:`~tracking.core.run.Run`
    objects.  One tracker is typically shared across the lifetime of a
    training session or simulation batch.

    Args:
        tracking_uri: Directory that will hold ``tracking.db`` and the
            ``artifacts/`` sub-tree.
        buffer_size: Number of metric points to buffer per run before
            flushing to SQLite.
    """

    def __init__(self, tracking_uri: str, buffer_size: int = 200) -> None:
        self.tracking_uri = tracking_uri
        self.buffer_size = buffer_size

        # Ensure base directory exists so SQLite can create the tracking.db file
        os.makedirs(tracking_uri, exist_ok=True)

        db_path = os.path.join(tracking_uri, "tracking.db")
        self._store = TrackingStore(db_path)

        self._artifacts_root = os.path.join(tracking_uri, "artifacts")
        os.makedirs(self._artifacts_root, exist_ok=True)

    # ------------------------------------------------------------------
    # Run creation
    # ------------------------------------------------------------------

    def start_run(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        run_type: str = "generic",
        tags: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> Run:
        """Create a new run and return it as a context manager.

        Example::

            with tracker.start_run("AM-VRPP-50", run_type="training") as run:
                run.log_params({"lr": 1e-4})
                trainer.fit(model)

        Args:
            experiment_name: Name of the parent experiment (created if absent).
            run_name: Optional human-readable run label.
            run_type: Semantic type: ``'training'``, ``'simulation'``,
                ``'evaluation'``, or ``'data_gen'``.
            tags: Key/value metadata to attach immediately.
            description: Free-text experiment description (first call only).

        Returns:
            A :class:`~tracking.core.run.Run` instance.
        """
        exp_id = self._store.get_or_create_experiment(experiment_name, description)
        run_id = str(uuid.uuid4())

        artifact_dir = os.path.join(self._artifacts_root, experiment_name, run_id)
        os.makedirs(artifact_dir, exist_ok=True)

        self._store.create_run(run_id, exp_id, run_name, run_type, artifact_dir)

        if tags:
            self._store.set_tags(run_id, {k: str(v) for k, v in tags.items()})

        run = Run(run_id, self._store, artifact_dir, self.buffer_size)
        return run

    def attach_run(self, run_id: str) -> Run:
        """Attach to an *existing* run and register it as the active run.

        Useful in worker sub-processes that share the same database but need
        to write under the parent process's run context.

        Args:
            run_id: UUID of the run created by the parent process.

        Returns:
            A :class:`~tracking.core.run.Run` that writes to
            the existing database record.

        Raises:
            ValueError: If *run_id* does not exist in the store.
        """
        run_data = self._store.get_run(run_id)
        if run_data is None:
            raise ValueError(f"Run '{run_id}' not found in tracking store at {self.tracking_uri!r}")
        artifact_dir = run_data.get("artifact_dir") or ""
        run = Run(run_id, self._store, artifact_dir, self.buffer_size)
        set_active_run(run)
        return run

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_experiments(self) -> List[Dict[str, Any]]:
        """Return all experiments ordered by creation time (newest first)."""
        return self._store.list_experiments()

    def list_runs(
        self,
        experiment_name: Optional[str] = None,
        run_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return runs, optionally filtered by experiment, type, or status."""
        exp_id: Optional[int] = None
        if experiment_name:
            for exp in self._store.list_experiments():
                if exp["name"] == experiment_name:
                    exp_id = int(exp["id"])
                    break
        return self._store.list_runs(experiment_id=exp_id, run_type=run_type, status=status)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Return the raw run record dict, or ``None`` if not found."""
        return self._store.get_run(run_id)
