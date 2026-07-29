"""DR-ALNS specific configuration."""

from dataclasses import dataclass


@dataclass
class DRALNSConfig:
    """DR-ALNS specific configuration."""

    max_iterations: int = 100
    n_destroy_ops: int = 3
    n_repair_ops: int = 2
    n_steps_per_epoch: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
