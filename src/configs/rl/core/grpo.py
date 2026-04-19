"""GRPO specific configuration."""

from dataclasses import dataclass


@dataclass
class GRPOConfig:
    """GRPO specific configuration."""

    group_size: int = 8
    epsilon: float = 0.2
    epochs: int = 3
