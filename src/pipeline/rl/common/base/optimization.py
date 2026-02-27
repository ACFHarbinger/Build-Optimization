"""
Optimization logic for RL4COLitModule.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


class OptimizationMixin:
    """Mixin for optimizer and scheduler configuration."""

    def __init__(self):
        """Initialize Class.

        Args:
            None.
        """
        # Type hints for attributes expected from the main class
        self.policy: Any
        self.optimizer_name: str
        self.optimizer_kwargs: dict
        self.lr_scheduler_name: Optional[str]
        self.lr_scheduler_kwargs: dict

    def configure_optimizers(self) -> Any:
        """Configure optimizer and optional scheduler."""
        optimizer: Any = None
        # Get optimizer
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(self.policy.parameters(), **self.optimizer_kwargs)
        elif self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.policy.parameters(), **self.optimizer_kwargs)
        elif self.optimizer_name.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop(self.policy.parameters(), **self.optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        opt: Any = optimizer

        if self.lr_scheduler_name is None:
            return optimizer

        scheduler: Any = None
        # Get scheduler
        name = self.lr_scheduler_name.lower()
        if name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, **self.lr_scheduler_kwargs)
        elif name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(opt, **self.lr_scheduler_kwargs)
        elif name == "lambda":
            gamma = self.lr_scheduler_kwargs.get("gamma", 1.0)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: gamma**epoch)
        elif name == "exp":
            gamma = self.lr_scheduler_kwargs.get("gamma", 0.99)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
        elif name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **self.lr_scheduler_kwargs)
        else:
            raise ValueError(f"Unknown scheduler: {self.lr_scheduler_name}")

        return {"optimizer": opt, "lr_scheduler": scheduler}
