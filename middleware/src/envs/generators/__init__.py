"""Synthetic problem-instance generators for build-optimisation environments."""

from typing import Any, Dict, Type

from .base import BuildGenerator
from .fps import FPSGenerator
from .moba import MOBAGenerator
from .rpg import RPGGenerator

GENERATOR_REGISTRY: Dict[str, Type[BuildGenerator]] = {
    "rpg": RPGGenerator,
    "fps": FPSGenerator,
    "moba": MOBAGenerator,
}


def get_generator(name: str, **kwargs: Any) -> BuildGenerator:
    """Factory: instantiate a generator by game-type name."""
    name = name.lower()
    if name == "rpg":
        return RPGGenerator(**kwargs)
    if name == "fps":
        return FPSGenerator(**kwargs)
    if name == "moba":
        return MOBAGenerator(**kwargs)
    raise ValueError(f"Unknown generator: '{name}'. Available: {list(GENERATOR_REGISTRY)}")


__all__ = [
    "BuildGenerator",
    "RPGGenerator",
    "FPSGenerator",
    "MOBAGenerator",
    "GENERATOR_REGISTRY",
    "get_generator",
]
