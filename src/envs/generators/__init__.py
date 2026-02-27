"""Synthetic problem-instance generators for build-optimisation environments."""

from .base import BuildGenerator
from .fps import FPSGenerator
from .moba import MOBAGenerator
from .rpg import RPGGenerator

GENERATOR_REGISTRY = {
    "rpg": RPGGenerator,
    "fps": FPSGenerator,
    "moba": MOBAGenerator,
}


def get_generator(name: str, **kwargs) -> BuildGenerator:
    """Factory: instantiate a generator by game-type name."""
    name = name.lower()
    if name not in GENERATOR_REGISTRY:
        raise ValueError(f"Unknown generator: '{name}'. Available: {list(GENERATOR_REGISTRY)}")
    return GENERATOR_REGISTRY[name](**kwargs)


__all__ = [
    "BuildGenerator",
    "RPGGenerator",
    "FPSGenerator",
    "MOBAGenerator",
    "GENERATOR_REGISTRY",
    "get_generator",
]
