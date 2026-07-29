"""
Configuration parameters for the Genetic Programming Multi-Period Hyper-Heuristic (GP-MP-HH).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class GP_MP_HH_Params:
    """
    Configuration parameters for GP-MP-HH.

    Attributes:
        pop_size: Size of the population.
        gens: Number of generations.
        prog_len: Length of the GP program.
        seed: Random seed.
    """

    pop_size: int = 10
    gens: int = 20
    prog_len: int = 5
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> GP_MP_HH_Params:
        """Create GP_MP_HH_Params from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            pop_size=getattr(config, "pop_size", 10),
            gens=getattr(config, "gens", 20),
            prog_len=getattr(config, "prog_len", 5),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert GPM_PHH_Params to a dictionary for backend compatibility."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
