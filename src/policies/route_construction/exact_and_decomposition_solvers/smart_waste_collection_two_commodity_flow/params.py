"""
Configuration parameters for the Smart Waste Collection - Two-Commodity Flow (SWC-TCF) policy.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class SWCTCFParams:
    """
    Configuration parameters for the SWC-TCF solver.

    Attributes:
        framework: Optimization framework to use ('ortools' or 'pyomo').
        engine: Optimization engine to use ('gurobi', 'scip', 'highs', or 'cplex').
        time_limit: Time limit for the solver in seconds.
        seed: Random seed for reproducibility.
    """

    framework: str = "ortools"
    engine: str = "gurobi"
    time_limit: float = 60.0
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> SWCTCFParams:
        """Create SWCTCFParams from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            framework=getattr(config, "framework", "ortools"),
            engine=getattr(config, "engine", "gurobi"),
            time_limit=float(getattr(config, "time_limit", 60.0)),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Params to a dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
