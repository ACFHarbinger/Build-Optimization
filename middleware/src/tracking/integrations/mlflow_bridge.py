"""MLflow secondary sink for WSTracker runs.

:class:`MLflowBridge` attaches to a :class:`~tracking.core.run.Run`
via :meth:`~tracking.core.run.Run.add_sink` and mirrors every
metric, parameter, and artifact write to an MLflow tracking server.  It acts
as a transparent secondary backend — primary WSTracker writes are unaffected
even if the MLflow server is unreachable.

Typical usage
-------------
::

    import mlflow
    import tracking as wst
    from tracking.integrations.mlflow_bridge import MLflowBridge

    tracker = wst.init(experiment_name="AM-VRPP-50")
    with tracker.start_run("AM-VRPP-50", run_type="training") as run:
        bridge = MLflowBridge.attach(
            run,
            mlflow_tracking_uri="http://localhost:5000",
            experiment_name="AM-VRPP-50",
            run_name=run.run_id[:8],
        )
        run.log_params({"lr": 1e-4})   # written to both WSTracker and MLflow
        run.log_metric("val/reward", 0.95, step=1)
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional

from tracking.core.run import Run


class MLflowBridge:
    """Forwards :class:`Run` events to an MLflow tracking server.

    This object satisfies the sink protocol expected by
    :meth:`~tracking.core.run.Run.add_sink`:

    * ``log_metric(key, value, step)``
    * ``log_params(params)``
    * ``log_artifact(path)``
    * ``finish(status)``

    All methods catch and suppress every exception so a broken or
    unreachable MLflow server never disrupts the training loop.

    Args:
        mlflow_tracking_uri: MLflow tracking server URI or local path.
            Passed to ``mlflow.set_tracking_uri()``.
        experiment_name: MLflow experiment name.
        run_name: Optional human-readable run name shown in the UI.
        tags: Extra key/value tags attached to the MLflow run.
    """

    def __init__(
        self,
        mlflow_tracking_uri: str,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        import mlflow

        self._mlflow = mlflow
        self._active_run: Optional[Any] = None

        with contextlib.suppress(Exception):
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._active_run = mlflow.start_run(
                run_name=run_name,
                tags=tags or {},
            )

    # ------------------------------------------------------------------
    # Sink protocol
    # ------------------------------------------------------------------

    def log_metric(self, key: str, value: float, step: int) -> None:
        """Forward a scalar metric to MLflow."""
        if self._active_run is None:
            return
        with contextlib.suppress(Exception):
            self._mlflow.log_metric(key, value, step=step)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Forward parameters to MLflow (values coerced to str)."""
        if self._active_run is None:
            return
        with contextlib.suppress(Exception):
            # MLflow param values must be str; split into chunks of 100
            str_params = {k: str(v) for k, v in params.items()}
            items = list(str_params.items())
            for i in range(0, len(items), 100):
                self._mlflow.log_params(dict(items[i : i + 100]))

    def log_artifact(self, path: str) -> None:
        """Forward a local file path as an MLflow artifact."""
        if self._active_run is None:
            return
        with contextlib.suppress(Exception):
            self._mlflow.log_artifact(path)

    def finish(self, status: str = "completed") -> None:
        """End the MLflow run, mapping WSTracker status to MLflow status."""
        if self._active_run is None:
            return
        with contextlib.suppress(Exception):
            mlflow_status = "FINISHED" if status == "completed" else "FAILED"
            self._mlflow.end_run(status=mlflow_status)
            self._active_run = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def attach(
        cls,
        run: Run,
        mlflow_tracking_uri: str,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> "MLflowBridge":
        """Create a bridge and immediately attach it to *run*.

        Args:
            run: The active :class:`Run` to mirror.
            mlflow_tracking_uri: MLflow server URI or local path.
            experiment_name: MLflow experiment name.
            run_name: Optional display name for the MLflow run.
            tags: Extra tags forwarded to the MLflow run.

        Returns:
            The created :class:`MLflowBridge` instance (already attached).
        """
        bridge = cls(
            mlflow_tracking_uri=mlflow_tracking_uri,
            experiment_name=experiment_name,
            run_name=run_name or run.run_id[:8],
            tags=tags,
        )
        run.add_sink(bridge)
        return bridge
