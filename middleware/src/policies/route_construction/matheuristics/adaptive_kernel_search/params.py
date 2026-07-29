"""
Configuration parameters for the Adaptive Kernel Search (AKS) matheuristic.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class AKSParams:
    """
    Configuration parameters for Adaptive Kernel Search.

    Attributes:
        initial_kernel_size: Size of the starting variable pool.
        bucket_size: Size for search buckets.
        max_buckets: Limit on improvement attempts.
        time_limit: Total time budget for optimization.
        mip_limit_nodes: Node limit for internal Gurobi solves.
        mip_gap: Acceptable relative optimality gap.
        t_easy: Threshold for 'EASY' instance classification (seconds).
        epsilon: Tolerance for fixing variables in 'HARD' instances.
        seed: Random seed.
    """

    initial_kernel_size: int = 50
    bucket_size: int = 20
    max_buckets: int = 15
    time_limit: float = 300.0
    mip_limit_nodes: int = 10000
    mip_gap: float = 0.01
    t_easy: float = 10.0
    epsilon: float = 0.1
    seed: int = 42
    engine: str = "gurobi"
    framework: str = "ortools"

    @classmethod
    def from_config(cls, config: Any) -> AKSParams:
        """Create AKSParams from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            initial_kernel_size=getattr(config, "initial_kernel_size", 50),
            bucket_size=getattr(config, "bucket_size", 20),
            max_buckets=getattr(config, "max_buckets", 15),
            time_limit=getattr(config, "time_limit", 300.0),
            mip_limit_nodes=getattr(config, "mip_limit_nodes", 10000),
            mip_gap=getattr(config, "mip_gap", 0.01),
            t_easy=getattr(config, "t_easy", 10.0),
            epsilon=getattr(config, "epsilon", 0.1),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert AKSParams to a dictionary for backend compatibility."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
