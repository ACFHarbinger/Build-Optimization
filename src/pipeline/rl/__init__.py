"""
Reinforcement Learning subpackage for WSmart-Route.

This package contains PyTorch Lightning modules, algorithms, and utilities
for training neural routing policies using reinforcement learning.

Core Modules:
- core/: Base RL modules, loss functions, and baselines
- meta/: Meta-learning and multi-objective optimization
- hpo/: Hyperparameter optimization utilities
- features/: Training features and hooks

Registries:
- ``RL_ALGORITHM_REGISTRY``: Maps algorithm names to Lightning module classes.
- ``get_rl_algorithm(name)``: Factory function for looking up algorithm classes.
"""

from logic.src.pipeline.rl.core import (
    A2C,
    DRGRPO,
    GDPO,
    GSPO,
    POMO,
    PPO,
    REINFORCE,
    SAPO,
    AdaptiveImitation,
    HRLModule,
    ImitationLearning,
    MetaRLModule,
    MVMoE_AM,
    MVMoE_POMO,
    SymNCO,
)

# RL Algorithm Registry: maps CLI algorithm names to Lightning module classes
RL_ALGORITHM_REGISTRY = {
    "reinforce": REINFORCE,
    "ppo": PPO,
    "a2c": A2C,
    "sapo": SAPO,
    "gspo": GSPO,
    "gdpo": GDPO,
    "dr_grpo": DRGRPO,
    "pomo": POMO,
    "symnco": SymNCO,
    "imitation": ImitationLearning,
    "adaptive_imitation": AdaptiveImitation,
    "hrl": HRLModule,
    "meta_rl": MetaRLModule,
    "mvmoe_pomo": MVMoE_POMO,
    "mvmoe_am": MVMoE_AM,
}


def get_rl_algorithm(name: str) -> type:
    """
    Look up an RL algorithm class by its short name.

    Args:
        name: Algorithm name (e.g. "reinforce", "ppo", "pomo").

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
    "A2C",
    "SAPO",
    "GSPO",
    "DRGRPO",
    "GDPO",
    "HRLModule",
    "MetaRLModule",
    "POMO",
    "SymNCO",
    "ImitationLearning",
    "AdaptiveImitation",
    "MVMoE_POMO",
    "MVMoE_AM",
    "RL_ALGORITHM_REGISTRY",
    "get_rl_algorithm",
]
