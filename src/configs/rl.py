"""
RL (Reinforcement Learning) configuration.
"""

from dataclasses import dataclass


@dataclass
class RLConfig:
    """Reinforcement Learning configuration.

    Attributes:
        algorithm: RL algorithm name (e.g. 'reinforce', 'ppo', 'a2c').
        baseline: Baseline type ('exponential', 'critic', 'rollout', None).
        gamma: Discount factor.
        entropy_coeff: Entropy bonus coefficient.
        max_grad_norm: Gradient clipping norm.
        n_epochs_inner: Inner PPO epochs per batch.
        wandb_mode: WandB mode ('online', 'offline', 'disabled').
        no_tensorboard: Whether to disable TensorBoard.
    """

    algorithm: str = "reinforce"
    baseline: str = "exponential"
    gamma: float = 0.99
    entropy_coeff: float = 0.01
    max_grad_norm: float = 1.0
    n_epochs_inner: int = 1
    wandb_mode: str = "disabled"
    no_tensorboard: bool = True
