"""Helper functions and utilities for the WSTracker Lightning integration."""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from tracking.core.run import get_active_run
from tracking.logging.pylogger import get_pylogger

if TYPE_CHECKING:
    from configs.tracking import TrackingConfig

logger = get_pylogger(__name__)


def _opt(cfg: Optional["TrackingConfig"], attr: str, default: Any = False) -> Any:
    """Read a flag from the tracking config, defaulting when absent."""
    if cfg is None:
        return default
    return getattr(cfg, attr, default)


# ---------------------------------------------------------------------------
# Hook Management
# ---------------------------------------------------------------------------


def register_monitoring_hooks(
    cfg: Optional["TrackingConfig"], pl_module: pl.LightningModule, hook_data: Dict[str, Any]
) -> None:
    """Register monitoring hooks based on tracking config flags."""
    try:
        from tracking.hooks import (
            add_activation_statistics_hook,
            add_gradient_monitoring_hooks,
            add_gradient_nan_detector_hook,
            add_weight_distribution_monitor,
        )
    except ImportError:
        logger.debug("Hooks module not available, skipping hook registration")
        return

    if _opt(cfg, "log_gradients"):
        try:
            hook_data["gradient"] = add_gradient_monitoring_hooks(pl_module)
        except Exception:
            logger.debug("Failed to register gradient hooks", exc_info=True)

    if _opt(cfg, "log_activations"):
        try:
            hook_data["activation"] = add_activation_statistics_hook(pl_module)
        except Exception:
            logger.debug("Failed to register activation hooks", exc_info=True)

    if _opt(cfg, "log_weights"):
        try:
            hook_data["weight"] = add_weight_distribution_monitor(pl_module)
        except Exception:
            logger.debug("Failed to register weight hooks", exc_info=True)

    if _opt(cfg, "nan_guard"):
        try:
            hook_data["nan_guard"] = add_gradient_nan_detector_hook(pl_module, raise_on_nan=True)
        except Exception:
            logger.debug("Failed to register NaN guard hooks", exc_info=True)


def log_hook_stats_to_run(run: Any, epoch: int, hook_data: Dict[str, Any]) -> None:
    """Forward accumulated hook statistics to WSTracker."""
    try:
        from tracking.hooks import (
            compute_activation_statistics,
            register_hooks_with_run,
        )
    except ImportError:
        return

    # Gradient stats
    if "gradient" in hook_data:
        with contextlib.suppress(Exception):
            register_hooks_with_run(hook_data["gradient"], run, prefix="train/hooks")

    # Activation stats
    if "activation" in hook_data:
        try:
            stats = compute_activation_statistics(hook_data["activation"].get("statistics", {}))
            for name, stat in stats.items():
                for k, v in stat.items():
                    if isinstance(v, (int, float)):
                        run.log_metric(f"train/act/{name}/{k}", float(v), step=epoch)
        except Exception:
            pass

    # Weight distribution stats
    if "weight" in hook_data:
        with contextlib.suppress(Exception):
            register_hooks_with_run(hook_data["weight"], run, prefix="train/hooks")


def remove_monitoring_hooks(hook_data: Dict[str, Any]) -> None:
    """Remove all registered monitoring hooks."""
    try:
        from tracking.hooks.activation_hooks import (
            remove_all_hooks as remove_act_hooks,
        )
        from tracking.hooks.gradient_hooks import (
            remove_all_hooks as remove_grad_hooks,
        )
    except ImportError:
        return

    for key in ["gradient", "nan_guard"]:
        if key in hook_data:
            with contextlib.suppress(Exception):
                remove_grad_hooks(hook_data[key])

    for key in ["activation"]:
        if key in hook_data:
            with contextlib.suppress(Exception):
                remove_act_hooks(hook_data[key])

    hook_data.clear()


# ---------------------------------------------------------------------------
# Visualisations & Profiling
# ---------------------------------------------------------------------------


def run_periodic_visualisations(
    cfg: Optional["TrackingConfig"],
    pl_module: pl.LightningModule,
    epoch: int,
) -> None:
    """Run configured visualisations (attention heatmaps, embeddings, loss landscape)."""
    log_dir = _opt(cfg, "log_dir", "logs")

    if _opt(cfg, "log_attention_heatmaps"):
        try:
            from tracking.logging.visualization.heatmaps import (
                plot_attention_heatmaps,
            )

            output_dir = os.path.join(log_dir, "attention_heatmaps")
            plot_attention_heatmaps(pl_module.policy, output_dir, epoch=epoch)

            run = get_active_run()
            if run is not None:
                run.log_artifact(output_dir, artifact_type="visualization")
        except Exception:
            logger.debug("Failed to plot attention heatmaps", exc_info=True)

    if _opt(cfg, "log_embeddings"):
        try:
            from tracking.logging.visualization.embeddings import (
                log_weight_distributions,
                project_node_embeddings,
            )

            tb_log_dir = os.path.join(log_dir, "tensorboard")
            log_weight_distributions(pl_module.policy, epoch, tb_log_dir)

            # Project embeddings if a validation batch is available
            if hasattr(pl_module, "val_dataset") and pl_module.val_dataset is not None:
                try:
                    import torch

                    val_loader = pl_module.val_dataloader()
                    x_batch = next(iter(val_loader))
                    if isinstance(x_batch, torch.Tensor):
                        pass  # skip if not dict-like
                    else:
                        project_node_embeddings(pl_module.policy, x_batch, tb_log_dir, epoch=epoch)
                except Exception:
                    logger.debug("Failed to project node embeddings", exc_info=True)
        except Exception:
            logger.debug("Failed to log embeddings/weight distributions", exc_info=True)

    if _opt(cfg, "log_loss_landscape"):
        try:
            from tracking.logging.visualization.landscape import (
                plot_loss_landscape,
            )

            output_dir = os.path.join(log_dir, "loss_landscape")
            # Requires root cfg on pl_module — safe access
            root_cfg = getattr(pl_module, "cfg", None)
            if root_cfg is not None and hasattr(pl_module, "policy"):
                plot_loss_landscape(
                    pl_module.policy,
                    root_cfg,
                    output_dir,
                    epoch=epoch,
                    size=20,
                    batch_size=8,
                    resolution=10,
                )
                run = get_active_run()
                if run is not None:
                    run.log_artifact(output_dir, artifact_type="visualization")
        except Exception:
            logger.debug("Failed to plot loss landscape", exc_info=True)


def log_execution_profiling_report() -> None:
    """Generate and log execution profiling report to WSTracker."""
    try:
        from tracking.profiling.profiler import _profiler_instance

        if _profiler_instance is not None:
            report = _profiler_instance.get_report()
            report.log_to_run()
    except Exception:
        logger.debug("Failed to log profiling report", exc_info=True)


# ---------------------------------------------------------------------------
# General Utilities
# ---------------------------------------------------------------------------


def extract_metrics(
    callback_metrics: Dict[str, Any],
    prefix: str,
) -> Dict[str, float]:
    """Extract metrics from Lightning callback_metrics that begin with prefix."""
    result: Dict[str, float] = {}
    for k, v in callback_metrics.items():
        if k.startswith(prefix):
            with contextlib.suppress(TypeError, ValueError):
                result[k] = float(v.item() if hasattr(v, "item") else v)
    return result


def log_checkpoint_artifact(
    run: Any,
    cb: ModelCheckpoint,
    epoch: int,
) -> None:
    """Log Lightning checkpoint paths directly as run artifacts."""
    for attr, label in [("best_model_path", "best_checkpoint"), ("last_model_path", "last_checkpoint")]:
        path: Optional[str] = getattr(cb, attr, None)
        if path and os.path.exists(path):
            run.log_artifact(
                path,
                name=label,
                artifact_type="checkpoint",
                metadata={
                    "epoch": epoch,
                    "monitor": cb.monitor,
                    "best": attr == "best_model_path",
                },
            )
