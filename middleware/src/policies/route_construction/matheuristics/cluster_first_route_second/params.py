"""
Configuration parameters for the Cluster-First Route-Second (CF-RS) policy.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class CFRSParams:
    """
    Configuration parameters for the CF-RS solver.

    Attributes:
        num_clusters: Number of clusters to form (None = auto).
        assignment_method: Method for assigning bins to vehicles ('angular', 'mip', 'greedy').
        route_optimizer: Solver for the intra-cluster TSP ('lkh', 'ortools', 'greedy').
        strict_fleet: Whether to strictly enforce vehicle count.
        seed_criterion: Strategy for selecting initial cluster seeds ('max_dist', 'spread', 'random').
        mip_objective: Objective to optimize in MIP assignment ('distance', 'balance', 'hybrid').
        time_limit: Time limit for assignment phase.
        seed: Random seed for reproducibility.
    """

    num_clusters: Optional[int] = None
    assignment_method: str = "angular"
    route_optimizer: str = "lkh"
    strict_fleet: bool = True
    seed_criterion: str = "max_dist"
    mip_objective: str = "distance"
    time_limit: float = 10.0
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> CFRSParams:
        """Create CFRSParams from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            num_clusters=getattr(config, "num_clusters", None),
            assignment_method=getattr(config, "assignment_method", "angular"),
            route_optimizer=getattr(config, "route_optimizer", "lkh"),
            strict_fleet=getattr(config, "strict_fleet", True),
            seed_criterion=getattr(config, "seed_criterion", "max_dist"),
            mip_objective=getattr(config, "mip_objective", "distance"),
            time_limit=getattr(config, "time_limit", 10.0),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Params to a dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
