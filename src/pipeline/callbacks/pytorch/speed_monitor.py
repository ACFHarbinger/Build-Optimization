"""Speed monitoring callback for Lightning."""

import time
from typing import Optional

import pytorch_lightning as L
from lightning.fabric.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.parsing import AttributeDict


class SpeedMonitor(Callback):
    """
    Monitor the speed of each step and each epoch.
    Logs intra-step (forward/backward) and inter-step (data loading) times.
    """

    def __init__(
        self,
        intra_step_time: bool = True,
        inter_step_time: bool = True,
        epoch_time: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize SpeedMonitor callback.

        Args:
            intra_step_time: Whether to log forward/backward time.
            inter_step_time: Whether to log data loading time.
            epoch_time: Whether to log epoch duration.
            verbose: Whether to print timing info to console.
        """
        super().__init__()
        self._log_stats = AttributeDict(
            {
                "intra_step_time": intra_step_time,
                "inter_step_time": inter_step_time,
                "epoch_time": epoch_time,
            }
        )
        self.verbose = verbose
        self._snap_intra_step_time: Optional[float] = None
        self._snap_inter_step_time: Optional[float] = None
        self._snap_epoch_time: Optional[float] = None

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Reset epoch time snapshot at training start."""
        self._snap_epoch_time = None

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Reset timing snapshots and capture epoch start time."""
        self._snap_intra_step_time = None
        self._snap_inter_step_time = None
        self._snap_epoch_time = time.time()

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Reset inter-step time snapshot for validation."""
        self._snap_inter_step_time = None

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Reset inter-step time snapshot for testing."""
        self._snap_inter_step_time = None

    @rank_zero_only
    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        *unused_args,
        **unused_kwargs,
    ) -> None:
        """Capture intra-step start time and log inter-step time."""
        if self._log_stats.intra_step_time:
            self._snap_intra_step_time = time.time()

        if not self._should_log(trainer):
            return

        logs: dict[str, float] = {}
        if self._log_stats.inter_step_time and self._snap_inter_step_time:
            # First log at beginning of second step
            logs["time/inter_step_ms"] = (time.time() - self._snap_inter_step_time) * 1000

        if trainer.logger is not None:
            trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        *unused_args,
        **unused_kwargs,
    ) -> None:
        """Capture inter-step start time and log intra-step time."""
        if self._log_stats.inter_step_time:
            self._snap_inter_step_time = time.time()

        if self.verbose and self._log_stats.intra_step_time and self._snap_intra_step_time:
            pl_module.print(f"time/intra_step_ms: {(time.time() - self._snap_intra_step_time) * 1000}")

        if not self._should_log(trainer):
            return

        logs: dict[str, float] = {}
        if self._log_stats.intra_step_time and self._snap_intra_step_time:
            logs["time/intra_step_ms"] = (time.time() - self._snap_intra_step_time) * 1000

        if trainer.logger is not None:
            trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Log epoch duration at end of training epoch."""
        logs: dict[str, float] = {}
        if self._log_stats.epoch_time and self._snap_epoch_time:
            logs["time/epoch_s"] = time.time() - self._snap_epoch_time
        if trainer.logger is not None:
            trainer.logger.log_metrics(logs, step=trainer.global_step)

    @staticmethod
    def _should_log(trainer) -> bool:
        """should log.

        Args:
            trainer (Any): Description of trainer.

        Returns:
            Any: Description of return value.
        """
        return (trainer.global_step + 1) % trainer.log_every_n_steps == 0 or trainer.should_stop
