"""
Custom PyTorch Lightning Trainer for WSmart-Route.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from logic.src.configs.tracking import TrackingConfig

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks import (
    Callback,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import Logger, WandbLogger

from logic.src.pipeline.callbacks import (
    ModelSummaryCallback,
    TrainingDisplayCallback,
)
from logic.src.tracking.integrations.lightning import TrackingCallback

try:
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.strategies import DDPStrategy
except ImportError:
    TensorBoardLogger = None  # type: ignore[assignment,misc]
    DDPStrategy = None  # type: ignore[assignment,misc]


class WSTrainer(pl.Trainer):
    """
    Extended Trainer with WSmart-Route specific features.

    Provides sensible defaults for:
    - Checkpointing
    - Progress bars
    - Logging
    """

    def __init__(
        self,
        max_epochs: int = 100,
        accelerator: str = "auto",
        devices: Union[int, str] = "auto",
        gradient_clip_val: float = 1.0,
        log_every_n_steps: int = 50,
        check_val_every_n_epoch: int = 1,
        project_name: str = "wsmart-route",
        experiment_name: Optional[str] = None,
        callbacks: Optional[list[Callback]] = None,
        logger: Optional[Union[Logger, bool]] = None,
        enable_progress_bar: bool = True,
        model_weights_path: Optional[str] = None,
        logs_dir: Optional[str] = None,
        # RL4CO-style optimizations
        precision: Any = "16-mixed",
        matmul_precision: str = "medium",
        disable_jit_profiling: bool = True,
        auto_ddp: bool = True,
        reload_dataloaders_every_n_epochs: int = 1,
        tracking_cfg: Optional["TrackingConfig"] = None,
        **kwargs,
    ):
        """
        Initialize WSTrainer with RL-specific optimizations.

        Args:
            max_epochs: Maximum number of training epochs.
            accelerator: Hardware accelerator ('auto', 'gpu', 'cpu').
            devices: Number of devices or 'auto'.
            gradient_clip_val: Gradient clipping value.
            log_every_n_steps: Logging frequency.
            check_val_every_n_epoch: Validation frequency.
            project_name: WandB project name.
            experiment_name: Experiment name for logging.
            callbacks: Custom callbacks.
            logger: Logger instance or False to disable.
            enable_progress_bar: Show progress bar.
            model_weights_path: Path for checkpoint saving.
            logs_dir: Directory for logs.
            precision: Training precision ('32', '16-mixed', 'bf16-mixed').
            matmul_precision: Matmul precision for Ampere+ GPUs ('highest', 'high', 'medium').
            disable_jit_profiling: Disable JIT profiling for memory optimization.
            auto_ddp: Auto-configure DDP for multi-GPU.
            reload_dataloaders_every_n_epochs: Reload dataloaders every N epochs (for RL).
            **kwargs: Additional Trainer arguments.
        """
        # RL4CO Optimization 1: Disable JIT profiling (memory optimization)
        if disable_jit_profiling:
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)

        # RL4CO Optimization 2: Set matmul precision for Ampere+ GPUs
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision(matmul_precision)

        # RL4CO Optimization 3: Auto DDP configuration
        strategy = kwargs.pop("strategy", None)
        if strategy is None and auto_ddp:
            n_devices = devices if isinstance(devices, int) else torch.cuda.device_count()
            if n_devices > 1:
                strategy = DDPStrategy(find_unused_parameters=True)

        if strategy is None:
            strategy = "auto"

        # Build logger
        if logger is None:
            logger = self._create_default_logger(project_name, experiment_name, logs_dir)

        # Store tracking config for callback creation
        self._tracking_cfg = tracking_cfg

        # Build callbacks
        callbacks = callbacks or []
        callbacks = self._add_default_callbacks(callbacks, enable_progress_bar, model_weights_path, logger)

        # Check if a custom progress bar callback exists
        # If we have a custom progress bar, we must enable Lightning's progress bar system
        has_custom_progress_bar = any(isinstance(c, RichProgressBar) for c in callbacks)
        lightning_enable_progress_bar = has_custom_progress_bar or enable_progress_bar
        super().__init__(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            gradient_clip_val=gradient_clip_val,
            log_every_n_steps=log_every_n_steps,
            check_val_every_n_epoch=check_val_every_n_epoch,
            callbacks=callbacks,
            logger=logger,
            precision=precision,
            strategy=strategy,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            enable_progress_bar=lightning_enable_progress_bar,
            enable_model_summary=False,  # Use custom ModelSummaryCallback instead
            **kwargs,
        )

    def _add_default_callbacks(
        self,
        callbacks: list[Callback],
        enable_progress_bar: bool,
        model_weights_path: Optional[str] = None,
        logger: Optional[Union[Logger, bool]] = None,
    ) -> list[Callback]:
        """Add default callbacks if not present."""
        callback_types = {type(c) for c in callbacks}

        # Add checkpoint callback
        if ModelCheckpoint not in callback_types:
            dirpath = model_weights_path if model_weights_path else "checkpoints"

            # Append version if available from logger
            if dirpath and logger and hasattr(logger, "version"):
                version = logger.version  # type: ignore[union-attr]
                if isinstance(version, int):
                    version = f"version_{version}"
                if isinstance(version, str):
                    dirpath = os.path.join(dirpath, version)

            callbacks.append(
                ModelCheckpoint(
                    dirpath=dirpath,
                    monitor="val/reward",
                    mode="max",
                    save_top_k=3,
                    filename="{epoch}-{val_reward:.4f}",
                    save_last=True,
                )
            )

        # Handle progress bar and chart callbacks
        if enable_progress_bar:
            # Find if TrainingDisplayCallback exist
            display_callback = None

            for callback in callbacks:
                if isinstance(callback, TrainingDisplayCallback):
                    display_callback = callback
                    break

            # If the new unified callback is not present, add it
            if display_callback is None:
                callbacks.append(TrainingDisplayCallback())

        # Add custom model summary callback
        if ModelSummaryCallback not in callback_types:
            callbacks.append(ModelSummaryCallback())

        # Add tracking callback (no-op when no active run is registered)
        if TrackingCallback not in callback_types:
            callbacks.append(TrackingCallback(tracking_cfg=getattr(self, "_tracking_cfg", None)))

        return callbacks

    def _create_default_logger(
        self,
        project_name: str,
        experiment_name: Optional[str],
        logs_dir: Optional[str] = None,
    ) -> Logger:
        """Create default WandB logger."""
        try:
            return WandbLogger(
                project=project_name,
                name=experiment_name,
                log_model=False,
            )
        except (ImportError, Exception) as e:
            # Fall back to TensorBoard if WandB not available or network fails
            # We don't use pl.logger here because it might not be fully setup, but loguru is safe
            logger.warning(f"WandB initialization failed, falling back to TensorBoard: {e}")
            return TensorBoardLogger(logs_dir or "logs", name="")
