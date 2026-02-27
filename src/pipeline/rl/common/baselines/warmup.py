"""
Gradual transition warmup baseline.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from tensordict import TensorDict
from torch import nn

from .base import Baseline
from .exponential import ExponentialBaseline


class WarmupBaseline(Baseline):
    """Gradual transition from ExponentialBaseline to target baseline."""

    def __init__(
        self,
        baseline: Baseline,
        warmup_epochs: int = 1,
        bl_warmup_epochs: Optional[int] = None,
        beta: float = 0.8,
        exp_beta: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize WarmupBaseline.

        Args:
            baseline: Target baseline to transition to.
            warmup_epochs: Number of epochs for warmup transition.
            bl_warmup_epochs: Alternative name for warmup_epochs.
            beta: Beta parameter for the warmup exponential baseline.
            exp_beta: Alternative name for beta.
        """
        super().__init__()
        self.baseline = baseline
        self.warmup_epochs = bl_warmup_epochs if bl_warmup_epochs is not None else warmup_epochs
        self.warmup_baseline = ExponentialBaseline(beta=beta, exp_beta=exp_beta)
        self.alpha = 0.0

    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:  # type: ignore[override]
        """
        Compute blended baseline value based on warmup progress.

        Args:
            td: TensorDict with environment state.
            reward: Current batch rewards.
            env: Environment.

        Returns:
            torch.Tensor: Blended baseline value.
        """
        if self.alpha >= 1.0:
            return self.baseline.eval(td, reward, env)
        if self.alpha <= 0.0:
            return self.warmup_baseline.eval(td, reward, env)

        v_target = self.baseline.eval(td, reward, env)
        v_warmup = self.warmup_baseline.eval(td, reward, env)
        return self.alpha * v_target + (1 - self.alpha) * v_warmup

    def unwrap_batch(self, batch: Any) -> Tuple[Any, Optional[torch.Tensor]]:
        """Unwrap the batch using the inner baseline.

        Args:
            batch: Wrapped batch.

        Returns:
            Tuple: Unwrapped batch data and optional baseline.
        """
        return self.baseline.unwrap_batch(batch)

    def epoch_callback(
        self,
        policy: nn.Module,
        epoch: int,
        val_dataset: Optional[Any] = None,
        env: Optional[Any] = None,
    ):
        """
        Update warmup alpha and call inner baseline callback.

        Args:
            policy: Current policy.
            epoch: Current epoch number.
            val_dataset: Validation dataset.
            env: Environment.
        """
        self.baseline.epoch_callback(policy, epoch, val_dataset, env)
        if epoch < self.warmup_epochs:
            self.alpha = (epoch + 1) / float(self.warmup_epochs)
        else:
            self.alpha = 1.0

    def get_learnable_parameters(self) -> list:
        """Get learnable parameters of the inner baseline."""
        return self.baseline.get_learnable_parameters()
