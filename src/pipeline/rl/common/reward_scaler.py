"""
Reward scaling utilities for RL training.

Provides running mean/variance normalization for advantage estimation,
using Welford's online algorithm for numerical stability.

Reference: RL4CO (https://github.com/ai4co/rl4co)
"""

from __future__ import annotations

from typing import Literal, Optional

import torch


class RewardScaler:
    """
    Reward/advantage scaler using running statistics.

    Uses Welford's online algorithm for numerically stable
    computation of running mean and variance.

    Args:
        scale: Scaling mode:
            - "norm": Normalize to zero mean, unit variance
            - "scale": Divide by standard deviation only
            - "none": No scaling
        running_momentum: Momentum for exponential moving average (0 = cumulative, 1 = instant)
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        scale: Literal["norm", "scale", "none"] = "norm",
        running_momentum: float = 0.0,
        eps: float = 1e-8,
    ):
        """
        Initialize RewardScaler.

        Args:
            scale: Scaling mode ('norm', 'scale', 'none').
            running_momentum: Momentum for stats updates.
            eps: Numerical stability constant.
        """
        self.scale = scale
        self.running_momentum = running_momentum
        self.eps = eps

        # Running statistics (Welford's algorithm)
        self._count: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0  # Sum of squared differences from mean

        # EMA statistics (alternative)
        self._ema_mean: Optional[torch.Tensor] = None
        self._ema_var: Optional[torch.Tensor] = None

    @property
    def mean(self) -> float:
        """Current running mean."""
        return self._mean

    @property
    def variance(self) -> float:
        """Current running variance."""
        if self._count < 2:
            return 1.0
        return self._m2 / self._count

    @property
    def std(self) -> float:
        """Current running standard deviation."""
        return max(self.variance**0.5, self.eps)

    def update(self, scores: torch.Tensor) -> None:
        """
        Update running statistics with new scores.

        Uses Welford's online algorithm for numerical stability.

        Args:
            scores: Tensor of scores/rewards
        """
        if self.running_momentum > 0:
            # Use EMA
            self._update_ema(scores)
        else:
            # Use Welford's cumulative algorithm
            self._update_welford(scores)

    def _update_welford(self, scores: torch.Tensor) -> None:
        """Update using Welford's online algorithm (vectorized batch update)."""
        scores_flat = scores.detach().float().view(-1)
        n_b = scores_flat.numel()
        if n_b == 0:
            return

        # Compute batch statistics
        mu_b = scores_flat.mean().item()
        m2_b = (scores_flat.var(correction=0) * n_b).item()

        if self._count == 0:
            self._count = n_b
            self._mean = mu_b
            self._m2 = m2_b
        else:
            n_a = self._count
            n = n_a + n_b
            delta = mu_b - self._mean

            self._mean += delta * (n_b / n)
            self._m2 += m2_b + (delta**2) * (n_a * n_b / n)
            self._count = n

    def _update_ema(self, scores: torch.Tensor) -> None:
        """Update using exponential moving average."""
        scores_flat = scores.detach().float().view(-1)
        batch_mean = scores_flat.mean()
        batch_var = scores_flat.var()

        if self._ema_mean is None:
            self._ema_mean = batch_mean
            self._ema_var = batch_var
        else:
            alpha = self.running_momentum
            self._ema_mean = alpha * batch_mean + (1 - alpha) * self._ema_mean
            self._ema_var = alpha * batch_var + (1 - alpha) * self._ema_var if self._ema_var is not None else batch_var

        # Sync with Welford stats for property access
        self._mean = self._ema_mean.item()
        self._m2 = (self._ema_var.item() if self._ema_var is not None else 0.0) * max(self._count, 1)

    def __call__(self, scores: torch.Tensor, update: bool = True) -> torch.Tensor:
        """
        Scale scores using running statistics.

        Args:
            scores: Input scores/rewards
            update: Whether to update running statistics

        Returns:
            Scaled scores
        """
        if update:
            self.update(scores)

        if self.scale == "none":
            return scores
        elif self.scale == "norm":
            return (scores - self.mean) / (self.std + self.eps)
        elif self.scale == "scale":
            return scores / (self.std + self.eps)
        else:
            raise ValueError(f"Unknown scale mode: {self.scale}")

    def reset(self) -> None:
        """Reset running statistics."""
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._ema_mean = None
        self._ema_var = None

    def state_dict(self) -> dict:
        """Get state dictionary for checkpointing."""
        return {
            "count": self._count,
            "mean": self._mean,
            "m2": self._m2,
            "scale": self.scale,
            "running_momentum": self.running_momentum,
            "eps": self.eps,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from dictionary."""
        self._count = state.get("count", 0)
        self._mean = state.get("mean", 0.0)
        self._m2 = state.get("m2", 0.0)
