"""PPO specific configuration."""

from dataclasses import dataclass


@dataclass
class PPOConfig:
    """PPO specific configuration."""

    epochs: int = 10
    eps_clip: float = 0.2
    value_loss_weight: float = 0.5
    mini_batch_size: float = 0.25
