"""
Configuration parameters for the Capacitated Vehicle Routing Problem (CVRP) policy.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class CVRPParams:
    """
    Configuration parameters for the CVRP solver.

    Attributes:
        engine: Optimization engine to use ('pyvrp' or 'ortools').
        time_limit: Time limit for the solver in seconds.
        seed: Random seed for reproducibility.
    """

    engine: str = "pyvrp"
    time_limit: float = 2.0
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> CVRPParams:
        """Create CVRPParams from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            engine=getattr(config, "engine", "pyvrp"),
            time_limit=getattr(config, "time_limit", 2.0),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Params to a dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
