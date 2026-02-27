"""
Reinforcement Learning subpackage for Build-Optimization.

This package contains PyTorch Lightning modules, algorithms, and utilities
for training neural policies using reinforcement learning.

Registries:
- ``RL_ALGORITHM_REGISTRY``: Maps algorithm names to Lightning module classes.
- ``get_rl_algorithm(name)``: Factory function for looking up algorithm classes.
"""

from pipeline.rl.core.ppo import PPO
from pipeline.rl.core.reinforce import REINFORCE

# RL Algorithm Registry: maps CLI algorithm names to Lightning module classes
RL_ALGORITHM_REGISTRY = {
    "reinforce": REINFORCE,
    "ppo": PPO,
}


def get_rl_algorithm(name: str) -> type:
    """
    Look up an RL algorithm class by its short name.

    Args:
        name: Algorithm name (e.g. "reinforce", "ppo").

    Returns:
        The Lightning module class (not instantiated).

    Raises:
        ValueError: If the name is not found in the registry.
    """
    if name not in RL_ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown RL algorithm: {name!r}. Available: {sorted(RL_ALGORITHM_REGISTRY.keys())}")
    return RL_ALGORITHM_REGISTRY[name]


__all__ = [
    "REINFORCE",
    "PPO",
    "RL_ALGORITHM_REGISTRY",
    "get_rl_algorithm",
]
