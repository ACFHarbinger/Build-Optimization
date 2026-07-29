"""
Gradient norm and weight distribution tracker for PyTorch Lightning training.

Logs per-layer gradient norms, global gradient norm, weight distributions,
and grad/weight ratios to W&B or TensorBoard to monitor for mode collapse,
vanishing gradients, or exploding gradients during neural routing model training.

Zero-interference design: all logging is guarded by ``log_freq`` / ``hist_freq``
parameters and does NOT modify any tensor states.

Usage (Hydra config):
    callbacks:
      - gradient_tracker:
          _target_: pipeline.callbacks.pytorch.gradient_tracker.GradientTrackerCallback
          log_freq: 50
          hist_freq: 200
          log_histograms: true
          prefix: "debug/"
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Set

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import Callback


class GradientTrackerCallback(Callback):
    """
    Tracks gradient norms and weight distributions per layer during training.

    Logs the following metrics to the active logger (W&B or TensorBoard):

    * ``{prefix}global_grad_norm`` — L2 norm across all tracked parameters.
    * ``{prefix}grad_norm/{layer}`` — Per-layer gradient L2 norm.
    * ``{prefix}grad_weight_ratio/{layer}`` — Ratio of gradient norm to weight
      norm (a stability diagnostic; large values suggest exploding gradients).
    * ``{prefix}weights/{layer}`` — Weight histogram (W&B / TensorBoard).
    * ``{prefix}gradients/{layer}`` — Gradient histogram (W&B / TensorBoard).

    Args:
        log_freq: Log scalar gradient norms every N global training steps. Default 50.
        hist_freq: Log weight/gradient histograms every N global steps. Default 200.
        norm_type: Norm type for gradient computation. Accepts 1, 2, or ``float("inf")``.
            Default 2.0.
        log_histograms: Whether to log weight and gradient histograms.
            Requires W&B or TensorBoard logger. Default True.
        include_bias: Whether to track bias parameters. Default False.
        layer_filter: Optional list of substrings. Only layers whose parameter
            name contains at least one substring are tracked. None = track all. Default None.
        prefix: Metric key prefix prepended to all logged keys. Default ``"debug/"``.
        max_grad_norm_alert: Emit a ``RuntimeWarning`` when the global gradient
            norm exceeds this value. Set to None to disable. Default 100.0.
    """

    def __init__(
        self,
        log_freq: int = 50,
        hist_freq: int = 200,
        norm_type: float = 2.0,
        log_histograms: bool = True,
        include_bias: bool = False,
        layer_filter: Optional[List[str]] = None,
        prefix: str = "debug/",
        max_grad_norm_alert: Optional[float] = 100.0,
    ) -> None:
        super().__init__()
        self.log_freq = log_freq
        self.hist_freq = hist_freq
        self.norm_type = norm_type
        self.log_histograms = log_histograms
        self.include_bias = include_bias
        self.layer_filter: Optional[List[str]] = layer_filter
        self.prefix = prefix
        self.max_grad_norm_alert = max_grad_norm_alert

        self._step: int = 0
        self._alerted_steps: Set[int] = set()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _should_track(self, name: str) -> bool:
        """Return True if this parameter name should be included in logging."""
        if not self.include_bias and name.endswith(".bias"):
            return False
        if self.layer_filter is None:
            return True
        return any(substr in name for substr in self.layer_filter)

    def _grad_norm(self, param: nn.Parameter) -> Optional[float]:
        """Compute gradient norm for one parameter tensor. Returns None if no grad."""
        if param.grad is None:
            return None
        if math.isinf(self.norm_type):
            return param.grad.data.abs().max().item()
        return param.grad.data.norm(self.norm_type).item()

    def _detect_logger(self, trainer: pl.Trainer) -> Optional[str]:
        """Return ``'wandb'``, ``'tensorboard'``, or None for the active logger."""
        try:
            from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
        except ImportError:
            return None
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                return "wandb"
            if isinstance(logger, TensorBoardLogger):
                return "tensorboard"
        return None

    # ------------------------------------------------------------------
    # Lightning lifecycle hooks
    # ------------------------------------------------------------------

    def on_after_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """
        Hook called immediately after ``loss.backward()``.

        Logs per-layer gradient norms and the global grad norm every
        ``log_freq`` steps. Does NOT clip or modify any gradients.
        """
        if not trainer.is_global_zero:
            return

        self._step = trainer.global_step
        if self._step % self.log_freq != 0:
            return

        log_dict: Dict[str, float] = {}
        global_norm_sq = 0.0
        n_tracked = 0

        for name, param in pl_module.named_parameters():
            if not self._should_track(name):
                continue
            g_norm = self._grad_norm(param)
            if g_norm is None:
                continue

            n_tracked += 1
            global_norm_sq += g_norm**2
            safe = name.replace(".", "/")
            log_dict[f"{self.prefix}grad_norm/{safe}"] = g_norm

            # Grad/weight ratio — stability diagnostic
            w_norm = param.data.norm(self.norm_type).item()
            if w_norm > 1e-8:
                log_dict[f"{self.prefix}grad_weight_ratio/{safe}"] = g_norm / w_norm

        if n_tracked == 0:
            return

        global_norm = math.sqrt(global_norm_sq)
        log_dict[f"{self.prefix}global_grad_norm"] = global_norm

        pl_module.log_dict(log_dict, on_step=True, on_epoch=False, prog_bar=False)

        # Emit warning on gradient explosion (once per step)
        if (
            self.max_grad_norm_alert is not None
            and global_norm > self.max_grad_norm_alert
            and self._step not in self._alerted_steps
        ):
            self._alerted_steps.add(self._step)
            warnings.warn(
                f"[GradientTracker] Step {self._step}: global grad norm "
                f"{global_norm:.2f} exceeds threshold {self.max_grad_norm_alert}. "
                "Consider adding gradient clipping (gradient_clip_val in Trainer).",
                RuntimeWarning,
                stacklevel=2,
            )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """
        Log weight and gradient histograms every ``hist_freq`` steps.

        Dispatches to W&B or TensorBoard depending on the active logger.
        """
        if not trainer.is_global_zero:
            return
        step = trainer.global_step
        if not self.log_histograms or step % self.hist_freq != 0:
            return

        logger_type = self._detect_logger(trainer)
        if logger_type == "wandb":
            self._log_histograms_wandb(pl_module, step)
        elif logger_type == "tensorboard":
            self._log_histograms_tensorboard(trainer, pl_module, step)

    # ------------------------------------------------------------------
    # Logger-specific histogram methods
    # ------------------------------------------------------------------

    def _log_histograms_wandb(
        self,
        pl_module: pl.LightningModule,
        step: int,
    ) -> None:
        """Send weight and gradient histograms to W&B."""
        try:
            import wandb  # type: ignore[import]
        except ImportError:
            return

        if wandb.run is None:
            return

        hist_data: Dict[str, Any] = {}
        for name, param in pl_module.named_parameters():
            if not self._should_track(name):
                continue
            safe = name.replace(".", "/")
            hist_data[f"{self.prefix}weights/{safe}"] = wandb.Histogram(param.data.detach().cpu().float().numpy())
            if param.grad is not None:
                hist_data[f"{self.prefix}gradients/{safe}"] = wandb.Histogram(param.grad.detach().cpu().float().numpy())

        wandb.log(hist_data, step=step)

    def _log_histograms_tensorboard(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        step: int,
    ) -> None:
        """Send weight and gradient histograms to TensorBoard."""
        try:
            from pytorch_lightning.loggers import TensorBoardLogger
        except ImportError:
            return

        for logger in trainer.loggers:
            if not isinstance(logger, TensorBoardLogger):
                continue
            writer = logger.experiment
            for name, param in pl_module.named_parameters():
                if not self._should_track(name):
                    continue
                safe = name.replace(".", "/")
                writer.add_histogram(
                    f"{self.prefix}weights/{safe}",
                    param.data.detach().cpu(),
                    global_step=step,
                )
                if param.grad is not None:
                    writer.add_histogram(
                        f"{self.prefix}gradients/{safe}",
                        param.grad.detach().cpu(),
                        global_step=step,
                    )
            break  # Only log to the first TensorBoard logger found
