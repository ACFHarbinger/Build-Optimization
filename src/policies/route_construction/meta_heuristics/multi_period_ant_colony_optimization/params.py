"""
Configuration parameters for the Multi-Period Ant Colony Optimization (MP-ACO).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class MP_ACO_Params:
    """
    Configuration parameters for MP-ACO.

    Attributes:
        n_ants: Number of ants in the colony.
        iters: Number of iterations.
        alpha: Influence of pheromone on path selection.
        beta: Influence of heuristic information on path selection.
        rho: Pheromone evaporation rate.
        seed: Random seed.
    """

    n_ants: int = 10
    iters: int = 50
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> MP_ACO_Params:
        """Create MP_ACO_Params from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            n_ants=getattr(config, "n_ants", 10),
            iters=getattr(config, "iters", 50),
            alpha=getattr(config, "alpha", 1.0),
            beta=getattr(config, "beta", 2.0),
            rho=getattr(config, "rho", 0.1),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert MP_ACO_Params to a dictionary for backend compatibility."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
