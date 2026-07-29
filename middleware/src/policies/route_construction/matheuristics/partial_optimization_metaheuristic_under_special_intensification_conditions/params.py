"""
Configuration parameters for the POPMUSIC matheuristic.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class POPMUSICParams:
    """
    Configuration parameters for POPMUSIC.

    Attributes:
        subproblem_size: Total number of routes per subproblem.
        max_iterations: Maximum number of iterations (None = until U empty).
        base_solver: Solver for subproblems ("fast_tsp", "hgs", "alns").
        base_solver_config: Configuration for base_solver.
        cluster_solver: Solver for initial clustering.
        cluster_solver_config: Configuration for cluster solver.
        initial_solver: Initial solution generation method ("greedy", "nearest_neighbor").
        seed: Random seed.
        vrpp: Whether to use expansion pool for VRPP.
        profit_aware_operators: Whether to use profit-aware operators.
        k_prox: Proximity network size (KDTree).
        seed_strategy: Seed selection strategy ("lifo", "fifo", "random").
    """

    subproblem_size: int = 3
    max_iterations: Optional[int] = None
    base_solver: str = "fast_tsp"
    base_solver_config: Optional[Any] = None
    cluster_solver: str = "fast_tsp"
    cluster_solver_config: Optional[Any] = None
    initial_solver: str = "nearest_neighbor"
    seed: int = 42
    vrpp: bool = True
    profit_aware_operators: bool = False
    k_prox: int = 10
    seed_strategy: str = "lifo"

    @classmethod
    def from_config(cls, config: Any) -> POPMUSICParams:
        """Create POPMUSICParams from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            subproblem_size=getattr(config, "subproblem_size", 3),
            max_iterations=getattr(config, "max_iterations", None),
            base_solver=getattr(config, "base_solver", "fast_tsp"),
            base_solver_config=getattr(config, "base_solver_config", None),
            cluster_solver=getattr(config, "cluster_solver", "fast_tsp"),
            cluster_solver_config=getattr(config, "cluster_solver_config", None),
            initial_solver=getattr(config, "initial_solver", "nearest_neighbor"),
            seed=getattr(config, "seed", 42),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            k_prox=getattr(config, "k_prox", 10),
            seed_strategy=getattr(config, "seed_strategy", "lifo"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert POPMUSICParams to a dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
