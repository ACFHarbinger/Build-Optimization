"""PyTorch Lightning callback that forwards all training events to WSTracker."""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from tracking.core.run import get_active_run
from tracking.helpers.lightning_helpers import (
    _opt,
    extract_metrics,
    log_checkpoint_artifact,
    log_execution_profiling_report,
    log_hook_stats_to_run,
    register_monitoring_hooks,
    remove_monitoring_hooks,
    run_periodic_visualisations,
)
from tracking.logging.pylogger import get_pylogger

if TYPE_CHECKING:
    from configs.tracking import TrackingConfig

logger = get_pylogger(__name__)


class TrackingCallback(Callback):
    """Lightning callback that logs training metrics and artifacts to WSTracker.

    Hooks into the Lightning trainer lifecycle to:

    * Log model hyperparameters at training start.
    * Log ``train/*`` metrics at the end of every training epoch.
    * Log ``val/*`` metrics at the end of every validation epoch.
    * Log ``time/*`` speed-monitor metrics at training end.
    * Register the best and final model checkpoints as run artifacts.
    * Record dataset regeneration events when dataloaders are reloaded.
    * Optionally register gradient / activation / weight / memory hooks.
    * Optionally track throughput.
    * Optionally run visualisations (attention heatmaps, embeddings,
      loss landscapes) every *N* epochs or at the end of training.

    This callback is added automatically by :class:`WSTrainer` and requires
    no user configuration.
    """

    def __init__(self, tracking_cfg: Optional["TrackingConfig"] = None) -> None:
        super().__init__()
        self._cfg = tracking_cfg
        # Hook storage dicts keyed by category
        self._hook_data: Dict[str, Any] = {}
        self._throughput: Any = None
        self._memory_tracker: Any = None

    # ------------------------------------------------------------------
    # Fit lifecycle
    # ------------------------------------------------------------------

    def on_fit_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        run = get_active_run()
        if run is None:
            return

        # --- Hyperparameters ---
        hparams: Dict[str, Any] = {}
        if hasattr(pl_module, "hparams"):
            with contextlib.suppress(Exception):
                hparams.update(dict(pl_module.hparams))

        hparams.update(
            {
                "trainer.max_epochs": trainer.max_epochs,
                "trainer.precision": str(trainer.precision),
                "trainer.gradient_clip_val": trainer.gradient_clip_val,
                "trainer.log_every_n_steps": getattr(trainer, "log_every_n_steps", 50),
            }
        )
        run.log_params(hparams)
        run.set_tag("status", "training")

        # --- Register hooks ---
        register_monitoring_hooks(self._cfg, pl_module, self._hook_data)

        # --- Throughput tracker ---
        if _opt(self._cfg, "log_throughput"):
            try:
                from tracking.profiling.throughput import ThroughputTracker

                self._throughput = ThroughputTracker(unit="samples")
                self._throughput.start()
            except Exception:
                logger.debug("Could not initialise ThroughputTracker", exc_info=True)

        # --- Background memory tracker ---
        if _opt(self._cfg, "log_memory"):
            try:
                from tracking.profiling.memory import MemoryTracker

                self._memory_tracker = MemoryTracker(tag="training")
                self._memory_tracker.start()
            except Exception:
                logger.debug("Could not initialise MemoryTracker", exc_info=True)

    # ------------------------------------------------------------------
    # Per-epoch hooks
    # ------------------------------------------------------------------

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Record throughput per batch."""
        if self._throughput is not None:
            batch_size = trainer.train_dataloader.batch_size if trainer.train_dataloader else 1  # type: ignore[union-attr]
            self._throughput.record(batch_size)

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        run = get_active_run()
        if run is None:
            return
        epoch = trainer.current_epoch
        metrics = extract_metrics(trainer.callback_metrics, prefix="train/")
        run.log_metrics(metrics, step=epoch)

        # --- Hook stats per epoch ---
        log_hook_stats_to_run(run, epoch, self._hook_data)

        # --- Memory snapshot ---
        if _opt(self._cfg, "log_memory"):
            with contextlib.suppress(Exception):
                from tracking.profiling.memory import MemorySnapshot

                MemorySnapshot.capture(tag=f"epoch_{epoch}", step=epoch)

        # --- Throughput ---
        if self._throughput is not None:
            with contextlib.suppress(Exception):
                self._throughput.log_to_run(step=epoch, prefix="train")

        # --- Periodic expensive visualisations ---
        viz_n = _opt(self._cfg, "viz_every_n_epochs", 0)
        if viz_n and viz_n > 0 and epoch > 0 and epoch % viz_n == 0:
            run_periodic_visualisations(self._cfg, pl_module, epoch)

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        run = get_active_run()
        if run is None:
            return
        epoch = trainer.current_epoch
        metrics = extract_metrics(trainer.callback_metrics, prefix="val/")
        run.log_metrics(metrics, step=epoch)

    # ------------------------------------------------------------------
    # Train end
    # ------------------------------------------------------------------

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        run = get_active_run()
        if run is None:
            return

        # Log SpeedMonitor timing metrics
        time_metrics = extract_metrics(trainer.callback_metrics, prefix="time/")
        if time_metrics:
            run.log_metrics(time_metrics, step=trainer.current_epoch)

        # Register best and last checkpoints
        for cb in cast(Any, trainer).callbacks:
            if isinstance(cb, ModelCheckpoint):
                log_checkpoint_artifact(run, cb, trainer.current_epoch)

        # --- End-of-training visualisations ---
        run_periodic_visualisations(self._cfg, pl_module, trainer.current_epoch)

        # --- Profiling report ---
        if _opt(self._cfg, "log_profiling_report"):
            log_execution_profiling_report()

        # --- Stop memory tracker ---
        if self._memory_tracker is not None:
            with contextlib.suppress(Exception):
                self._memory_tracker.stop()
                self._memory_tracker.log_summary_to_run(step=trainer.current_epoch)

        # --- Throughput final log ---
        if self._throughput is not None:
            with contextlib.suppress(Exception):
                self._throughput.log_to_run(step=trainer.current_epoch, prefix="train")

        # --- Remove hooks ---
        remove_monitoring_hooks(self._hook_data)

        run.set_tag("status", "completed")
        run.set_tag("final_epoch", str(trainer.current_epoch))
        run.flush()

    # ------------------------------------------------------------------
    # Gradient / LR logging (existing)
    # ------------------------------------------------------------------

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Log gradient L2 norm and current learning rate."""
        run = get_active_run()
        if run is None:
            return
        step = trainer.global_step

        # Gradient L2 norm
        total_norm_sq = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
        run.log_metric("train/grad_norm", total_norm_sq**0.5, step=step)

        # Learning rate (first param group)
        for i, pg in enumerate(optimizer.param_groups):
            key = "train/lr" if i == 0 else f"train/lr_group{i}"
            run.log_metric(key, pg["lr"], step=step)

    # ------------------------------------------------------------------
    # Checkpoint events (fires after each ModelCheckpoint save)
    # ------------------------------------------------------------------

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        run = get_active_run()
        if run is None:
            return
        val_reward = trainer.callback_metrics.get("val/reward")
        epoch = trainer.current_epoch
        for cb in cast(Any, trainer).callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
                path = cb.best_model_path
                if os.path.exists(path):
                    run.log_artifact(
                        path,
                        artifact_type="checkpoint",
                        metadata={
                            "epoch": epoch,
                            "val_reward": float(val_reward) if val_reward is not None else None,
                        },
                    )

    # ------------------------------------------------------------------
    # Dataset regeneration detection
    # ------------------------------------------------------------------

    def on_train_dataloader(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Record a dataset mutation event when the dataloader is (re)loaded."""
        run = get_active_run()
        if run is None:
            return
        epoch = trainer.current_epoch
        if epoch == 0:
            return  # Skip initial load; it's already covered by log_dataset_event in the engine
        run.log_dataset_event(
            "mutate",
            metadata={
                "description": "Epoch dataloader reload",
                "epoch": epoch,
            },
        )
