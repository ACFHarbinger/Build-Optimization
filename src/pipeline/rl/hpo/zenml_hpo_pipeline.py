"""ZenML HPO pipeline for WSmart-Route.

Wraps :func:`~logic.src.pipeline.features.train.hpo.run_hpo` in a
three-step ZenML pipeline:

1. **prepare_hpo_config** — serialise the Hydra config to a plain dict.
2. **run_hpo_step** — reconstruct the config and delegate to
   :func:`run_hpo` with ZenML tracking disabled (to prevent re-dispatch).
3. **log_hpo_summary** — record the best metric as a ZenML artifact.

The pipeline is invoked from
:func:`~logic.src.pipeline.features.train.hpo._run_hpo_via_zenml`
when ``cfg.tracking.zenml_enabled`` is ``True``.
"""

from __future__ import annotations

from typing import Any, Dict

from omegaconf import OmegaConf

from logic.src.configs import Config
from logic.src.pipeline.features.train.hpo import run_hpo
from logic.src.tracking.logging.pylogger import get_pylogger

_ZENML_AVAILABLE = False
try:
    from zenml import pipeline as zenml_pipeline
    from zenml import step

    _ZENML_AVAILABLE = True
except ImportError:
    zenml_pipeline = None  # type: ignore[assignment,misc]
    step = None  # type: ignore[assignment,misc]

logger = get_pylogger(__name__)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

if _ZENML_AVAILABLE:

    @step  # type: ignore[misc]
    def prepare_hpo_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Pass-through: makes the serialised config a ZenML artifact."""
        return config_dict

    @step(experiment_tracker="mlflow_tracker")  # type: ignore[misc]
    def run_hpo_step(config_dict: Dict[str, Any]) -> float:
        """Execute HPO inside a ZenML-managed MLflow run.

        ZenML tracking is disabled in the reconstructed config to prevent
        the inner :func:`run_hpo` from re-dispatching to this pipeline.
        """
        cfg = OmegaConf.structured(Config)
        cfg = OmegaConf.merge(cfg, OmegaConf.create(config_dict))
        cfg = OmegaConf.to_object(cfg)
        assert isinstance(cfg, Config)

        # Prevent re-dispatch into ZenML
        tracking = getattr(cfg, "tracking", None)
        if tracking is not None:
            tracking.zenml_enabled = False  # type: ignore[union-attr]

        return run_hpo(cfg)

    @step  # type: ignore[misc]
    def log_hpo_summary(best_val: float) -> float:
        """Record the best HPO metric as a ZenML artifact."""
        logger.info(f"HPO complete — best metric: {best_val:.4f}")
        return best_val

    # -----------------------------------------------------------------------
    # Pipeline
    # -----------------------------------------------------------------------

    @zenml_pipeline(name="wsmart_route_hpo")  # type: ignore[misc]
    def _hpo_pipeline(config_dict: Dict[str, Any]) -> float:
        """Three-step HPO pipeline."""
        cfg_art = prepare_hpo_config(config_dict)
        best_val = run_hpo_step(cfg_art)
        return log_hpo_summary(best_val)


# ---------------------------------------------------------------------------
# Public entry point (called from hpo.py)
# ---------------------------------------------------------------------------


def hpo_pipeline(cfg: Any) -> float:
    """Serialise *cfg* and launch the ZenML HPO pipeline.

    Args:
        cfg: Root Hydra :class:`~logic.src.configs.Config` object.

    Returns:
        Best metric value found.

    Raises:
        ImportError: If ZenML is not installed.
    """
    if not _ZENML_AVAILABLE:
        raise ImportError("zenml is not installed — cannot run ZenML HPO pipeline")

    config_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    return _hpo_pipeline(config_dict=config_dict)  # type: ignore[return-value]
