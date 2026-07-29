"""
Activation function configuration module.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ActivationConfig:
    """Configuration for activation functions."""

    name: str = "gelu"
    param: float = 1.0
    threshold: float = 6.0
    replacement_value: float = 6.0
    n_params: int = 3
    # Use default_factory for mutable defaults
    range: List[float] = field(default_factory=lambda: [0.125, 1 / 3])
