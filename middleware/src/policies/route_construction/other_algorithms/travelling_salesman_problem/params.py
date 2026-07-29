"""
Configuration parameters for the Traveling Salesman Problem (TSP) policy.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class TSPParams:
    """
    Configuration parameters for the TSP solver.

    Attributes:
        time_limit: Time limit for the solver in seconds.
        seed: Random seed for reproducibility.
    """

    time_limit: float = 2.0
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> TSPParams:
        """Create TSPParams from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            time_limit=getattr(config, "time_limit", 2.0),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Params to a dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
