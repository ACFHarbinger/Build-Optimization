"""
Configuration parameters for the Multi-Period Iterated Local Search (MP-ILS).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class MP_ILS_Params:
    """
    Configuration parameters for MP-ILS.

    Attributes:
        iters: Number of iterations.
        perturb_size: Number of nodes to remove during perturbation.
        seed: Random seed.
    """

    iters: int = 50
    perturb_size: int = 3
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> MP_ILS_Params:
        """Create MP_ILS_Params from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            iters=getattr(config, "iters", getattr(config, "max_iter", 50)),
            perturb_size=getattr(config, "perturb_size", 3),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert MP_ILS_Params to a dictionary for backend compatibility."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
