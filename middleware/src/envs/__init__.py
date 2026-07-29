"""
Build-optimisation RL environments.

Mirrors the WSmart-Route env module structure but targets the sequential
item-selection (build optimisation) domain instead of vehicle routing.

Each game type is a separate environment class that shares the same
TorchRL-compatible interface (reset / step / action_mask) but uses
game-appropriate item stats, slot layouts, and scoring weights.

Environment registry
--------------------
    "rpg"   — Fantasy RPG  (10 slots, 7 stats: attack/defense/magic_power/…)
    "fps"   — Tactical FPS  (8 slots, 7 stats: damage/accuracy/range/…)
    "moba"  — MOBA          (6 slots, 8 stats: AD/AP/armor/MR/health/…)

Usage
-----
    from envs import get_env, RPGBuildEnv

    # Factory
    env = get_env("rpg", items_per_slot=6)
    td  = env.reset()
    td["action"] = td["action_mask"].float().multinomial(1).squeeze(-1)
    td  = env.step(td)["next"]

    # Direct
    env = RPGBuildEnv(batch_size=32)
"""

from envs.base import BuildEnvBase
from envs.fps import FPSBuildEnv
from envs.generators import (
    GENERATOR_REGISTRY,
    BuildGenerator,
    FPSGenerator,
    MOBAGenerator,
    RPGGenerator,
    get_generator,
)
from envs.moba import MOBABuildEnv
from envs.rpg import RPGBuildEnv

ENV_REGISTRY = {
    "rpg": RPGBuildEnv,
    "fps": FPSBuildEnv,
    "moba": MOBABuildEnv,
}


def get_env(name: str, **kwargs) -> BuildEnvBase:
    """
    Instantiate a build-optimisation environment by game-type name.

    Args:
        name:    Environment key — one of "rpg", "fps", "moba".
        **kwargs: Forwarded to the environment constructor.

    Returns:
        Initialised :class:`BuildEnvBase` subclass instance.

    Raises:
        ValueError: If *name* is not in the registry.
    """
    name = name.lower()
    if name not in ENV_REGISTRY:
        raise ValueError(f"Unknown environment: '{name}'. Available: {list(ENV_REGISTRY)}")
    return ENV_REGISTRY[name](**kwargs)


__all__ = [
    # Base
    "BuildEnvBase",
    # Generators
    "BuildGenerator",
    "RPGGenerator",
    "FPSGenerator",
    "MOBAGenerator",
    "get_generator",
    "GENERATOR_REGISTRY",
    # Environments
    "RPGBuildEnv",
    "FPSBuildEnv",
    "MOBABuildEnv",
    # Registry
    "ENV_REGISTRY",
    "get_env",
]
