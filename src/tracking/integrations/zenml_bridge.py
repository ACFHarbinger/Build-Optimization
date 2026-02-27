"""ZenML secondary sink and stack helpers for WSTracker runs.

:class:`ZenMLBridge` attaches to a :class:`~tracking.core.run.Run`
via :meth:`~tracking.core.run.Run.add_sink` and mirrors every metric,
parameter, and artifact write to the ZenML-managed MLflow experiment tracker.

Unlike :class:`MLflowBridge`, this class does **not** configure MLflow itself
(no ``set_tracking_uri``, ``set_experiment``, or ``start_run`` calls).  It
assumes that ZenML's stack has already started an MLflow run for the current
step.  All methods are silent no-ops when ``mlflow.active_run()`` is ``None``.

Typical usage (inside a ZenML step)
-------------------------------------
::

    from zenml import step
    from tracking.integrations.zenml_bridge import ZenMLBridge

    @step(experiment_tracker="mlflow_tracker")
    def run_training_step(config_dict: Dict[str, Any]) -> float:
        bridge = ZenMLBridge()
        val_reward = run_training(cfg, sinks=[bridge])
        return val_reward

Module-level helpers
----------------------
* :func:`configure_zenml_stack` — register an MLflow experiment tracker and
  create a named local ZenML stack programmatically.
* :func:`extract_zenml_step_output` — retrieve a step artifact from the last
  pipeline run via the ZenML client.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional

from tracking.core.run import Run
from tracking.logging.pylogger import get_pylogger

logger = get_pylogger(__name__)


class ZenMLBridge:
    """Forwards :class:`Run` events to the ZenML-managed MLflow experiment tracker.

    This sink is a transparent no-op when no MLflow run is active (i.e. when
    called outside a ZenML step decorated with
    ``experiment_tracker="mlflow_tracker"``).  ZenML owns the MLflow run
    lifecycle — ``finish()`` is therefore an explicit no-op on this class.

    The sink protocol implemented here mirrors :class:`MLflowBridge`:

    * ``log_metric(key, value, step)``
    * ``log_params(params)``
    * ``log_artifact(path)``
    * ``finish(status)`` — **no-op**
    """

    def __init__(self) -> None:
        self._mlflow: Optional[Any] = None
        with contextlib.suppress(ImportError):
            import mlflow  # type: ignore[import-not-found]

            self._mlflow = mlflow

    # ------------------------------------------------------------------
    # Internal guard
    # ------------------------------------------------------------------

    def _has_active_run(self) -> bool:
        """Return True only when an MLflow run is currently active."""
        if self._mlflow is None:
            return False
        with contextlib.suppress(Exception):
            return self._mlflow.active_run() is not None
        return False

    # ------------------------------------------------------------------
    # Sink protocol
    # ------------------------------------------------------------------

    def log_metric(self, key: str, value: float, step: int) -> None:
        """Forward a scalar metric to the active ZenML-managed MLflow run."""
        if not self._has_active_run():
            return
        with contextlib.suppress(Exception):
            self._mlflow.log_metric(key, value, step=step)  # type: ignore[union-attr]

    def log_params(self, params: Dict[str, Any]) -> None:
        """Forward parameters to the active MLflow run (values coerced to str)."""
        if not self._has_active_run():
            return
        with contextlib.suppress(Exception):
            str_params = {k: str(v) for k, v in params.items()}
            items = list(str_params.items())
            for i in range(0, len(items), 100):
                self._mlflow.log_params(dict(items[i : i + 100]))  # type: ignore[union-attr]

    def log_artifact(self, path: str) -> None:
        """Forward a local file path as an MLflow artifact."""
        if not self._has_active_run():
            return
        with contextlib.suppress(Exception):
            self._mlflow.log_artifact(path)  # type: ignore[union-attr]

    def finish(self, status: str = "completed") -> None:
        """No-op: ZenML ends the MLflow run when the step exits."""
        # Intentionally empty — calling mlflow.end_run() here would terminate
        # ZenML's experiment tracking run prematurely.

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def attach(cls, run: Run) -> "ZenMLBridge":
        """Create a :class:`ZenMLBridge` and attach it to *run*.

        Args:
            run: The active :class:`Run` to mirror into ZenML's MLflow tracker.

        Returns:
            The created :class:`ZenMLBridge` instance (already attached).
        """
        bridge = cls()
        run.add_sink(bridge)
        return bridge


# ---------------------------------------------------------------------------
# ZenML stack helpers
# ---------------------------------------------------------------------------


def configure_zenml_stack(
    mlflow_tracking_uri: str,
    stack_name: str = "wsmart-route-stack",
) -> bool:
    """Register an MLflow experiment tracker and create a local ZenML stack.

    Programmatically sets up a ZenML stack named *stack_name* that contains:

    * The default local artifact store.
    * The default local orchestrator.
    * An MLflow experiment tracker pointed at *mlflow_tracking_uri*.

    If the stack or its components already exist the function silently reuses
    them.  All :class:`zenml.client.Client` calls are wrapped in
    :func:`contextlib.suppress` so a missing or misconfigured ZenML
    installation never raises.

    Args:
        mlflow_tracking_uri: URI for the MLflow tracking server (local path,
            ``http://`` address, or ``databricks``).
        stack_name: Name for the ZenML stack to create or reuse.

    Returns:
        ``True`` if the stack was configured and activated successfully,
        ``False`` on any error.
    """
    try:
        from zenml.client import Client
        from zenml.enums import StackComponentType  # Import the enum

        client = Client()
        tracker_component_name = f"mlflow-tracker-{stack_name}"

        # Register the MLflow experiment tracker component
        with contextlib.suppress(Exception):
            client.create_stack_component(
                name=tracker_component_name,
                # Use the Enum member instead of the literal string
                component_type=StackComponentType.EXPERIMENT_TRACKER,
                flavor="mlflow",
                configuration={"tracking_uri": mlflow_tracking_uri},
            )

        # Create the stack
        with contextlib.suppress(Exception):
            client.create_stack(
                name=stack_name,
                components={StackComponentType.EXPERIMENT_TRACKER: tracker_component_name},
            )

        client.activate_stack(stack_name)
        return True

    except Exception as exc:
        logger.warning(f"ZenML stack configuration failed (non-fatal): {exc}")
        return False


def extract_zenml_step_output(
    pipeline_name: str,
    step_name: str,
) -> Optional[Any]:
    """Read a step artifact from the most recent ZenML pipeline run.

    Args:
        pipeline_name: Name of the ZenML pipeline (as registered in the store).
        step_name: Name of the step whose primary output to load.

    Returns:
        The materialised Python object, or ``None`` on any error.
    """
    with contextlib.suppress(Exception):
        from zenml.client import Client  # type: ignore[import-not-found]

        client = Client()
        last_run = client.get_pipeline(pipeline_name).last_run
        return last_run.steps[step_name].output.load()
    return None
