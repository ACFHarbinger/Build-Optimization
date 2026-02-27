"""
Optim Config module.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class OptimConfig:
    """Optimizer configuration.

    Attributes:
        optimizer: Name of the optimizer ('adam', 'sgd', etc.).
        lr: Learning rate.
        weight_decay: Weight decay factor.
        lr_scheduler: Name of the learning rate scheduler.
        lr_scheduler_kwargs: Keyword arguments for the LR scheduler.
    """

    optimizer: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 0.0
    lr_scheduler: Optional[str] = None
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Learning rate details
    lr_critic: float = 1e-4
    lr_decay: float = 1.0
    lr_min_value: float = 0.0
    lr_min_decay: float = 1e-8
