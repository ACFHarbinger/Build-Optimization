"""
Configuration parameters for the Population Hyper-Heuristic (PHH).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class PHHParams:
    """
    Configuration parameters for PHH.

    Attributes:
        pop_size: Size of the population.
        gens: Number of generations.
        seed: Random seed.
    """

    pop_size: int = 10
    gens: int = 20
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> PHHParams:
        """Create PHHParams from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            pop_size=getattr(config, "pop_size", 10),
            gens=getattr(config, "gens", 20),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert PHHParams to a dictionary for backend compatibility."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
