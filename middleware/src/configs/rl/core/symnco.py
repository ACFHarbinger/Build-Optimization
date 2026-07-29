"""SymNCO specific configuration."""

from dataclasses import dataclass


@dataclass
class SymNCOConfig:
    """SymNCO specific configuration."""

    alpha: float = 0.2
    beta: float = 1.0
