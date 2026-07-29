"""
Configuration parameters for the Branch-and-Price (BP) solver.

Standard Column Generation and Branch-and-Bound parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional


@dataclass
class BPParams:
    """
    Configuration parameters for the Branch-and-Price solver.

    Attributes:
        max_iterations: Maximum iterations for column generation loop.
        max_routes_per_iteration: Maximum columns to add per pricing call.
        optimality_gap: Relative gap for proven optimality.
        branching_strategy: Branching rule ('ryan_foster', 'edge', or 'divergence').
        max_branch_nodes: Maximum nodes to explore in the B&B tree.
        use_exact_pricing: Whether to use DP-based exact pricing (True)
            or a heuristic pricer (False).
        use_ng_routes: Whether to use ng-route relaxation in exact pricing.
        ng_neighborhood_size: Size of ng-neighborhoods for relaxation.
        tree_search_strategy: B&B search strategy ('best_first', 'depth_first').
        vehicle_limit: Maximum number of vehicles available.
        cleanup_frequency: Iterations between column cleanup rounds.
        cleanup_threshold: Minimum reduced cost for column retention.
        early_termination_gap: Gap at which to stop search early.
        multiple_waste_types: Whether the problem has multiple waste types.
        allow_heuristic_ryan_foster: Whether to allow Ryan-Foster branching
            even if it's theoretically a heuristic for the formulation.
    """

    max_iterations: int = 100
    max_routes_per_iteration: int = 10
    optimality_gap: float = 1e-4
    branching_strategy: str = "edge"
    max_branch_nodes: int = 1000
    use_exact_pricing: bool = False
    use_ng_routes: bool = True
    ng_neighborhood_size: int = 8
    tree_search_strategy: str = "best_first"
    vehicle_limit: Optional[int] = None
    cleanup_frequency: int = 20
    cleanup_threshold: float = -100.0
    early_termination_gap: float = 1e-3
    multiple_waste_types: bool = False
    allow_heuristic_ryan_foster: bool = False
    use_ryan_foster: bool = False
    # Multi-period adaptation (in essence solving the Inventory Routing Problem)
    multi_day_mode: bool = False

    @classmethod
    def from_config(cls, config: Any) -> BPParams:
        """Create BPParams from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            max_iterations=getattr(config, "max_iterations", 100),
            max_routes_per_iteration=getattr(config, "max_routes_per_iteration", 10),
            optimality_gap=getattr(config, "optimality_gap", 1e-4),
            branching_strategy=getattr(config, "branching_strategy", "edge"),
            max_branch_nodes=getattr(config, "max_branch_nodes", 1000),
            use_exact_pricing=getattr(config, "use_exact_pricing", False),
            use_ng_routes=getattr(config, "use_ng_routes", True),
            ng_neighborhood_size=getattr(config, "ng_neighborhood_size", 8),
            tree_search_strategy=getattr(config, "tree_search_strategy", "best_first"),
            vehicle_limit=getattr(config, "vehicle_limit", None),
            cleanup_frequency=getattr(config, "cleanup_frequency", 20),
            cleanup_threshold=getattr(config, "cleanup_threshold", -100.0),
            early_termination_gap=getattr(config, "early_termination_gap", 1e-3),
            multiple_waste_types=getattr(config, "multiple_waste_types", False),
            allow_heuristic_ryan_foster=getattr(config, "allow_heuristic_ryan_foster", False),
        )
