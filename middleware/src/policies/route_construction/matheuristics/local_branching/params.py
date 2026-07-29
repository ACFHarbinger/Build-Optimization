"""
Configuration parameters for the Local Branching (LB) policy.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class LBParams:
    """
    Configuration parameters for the Local Branching solver.

    Attributes:
        k: Hamming distance for the neighborhood (k-neighborhood).
        max_iterations: Maximum number of local branching iterations.
        time_limit: Total time limit for the entire optimization.
        time_limit_per_iteration: Time limit for each individual neighborhood search.
        node_limit_per_iteration: Gurobi node limit per neighborhood search.
        mip_gap: Target optimality gap for each sub-problem.
        seed: Random seed for reproducibility.
    """

    k: int = 10
    max_iterations: int = 100
    time_limit: float = 60.0
    time_limit_per_iteration: float = 10.0
    node_limit_per_iteration: int = 1000
    mip_gap: float = 0.01
    seed: int = 42
    engine: str = "gurobi"
    framework: str = "ortools"

    @classmethod
    def from_config(cls, config: Any) -> LBParams:
        """Create LBParams from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            k=getattr(config, "k", 10),
            max_iterations=getattr(config, "max_iterations", 100),
            time_limit=getattr(config, "time_limit", 60.0),
            time_limit_per_iteration=getattr(config, "time_limit_per_iteration", 10.0),
            node_limit_per_iteration=getattr(config, "node_limit_per_iteration", 1000),
            mip_gap=getattr(config, "mip_gap", 0.01),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Params to a dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
