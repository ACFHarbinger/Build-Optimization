"""
ALNS Policy Adapter.

Adapts the Adaptive Large Neighborhood Search (ALNS) logic to the agnostic interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import ALNSConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.adaptive_large_neighborhood_search.alns import run_alns

from .factory import PolicyRegistry


@PolicyRegistry.register("alns")
class ALNSPolicy(BaseRoutingPolicy):
    """
    ALNS policy class.

    Visits pre-selected 'must_go' bins using Adaptive Large Neighborhood Search.
    """

    def __init__(self, config: Optional[Union[ALNSConfig, Dict[str, Any]]] = None):
        """Initialize ALNS policy with optional config.

        Args:
            config: ALNSConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return ALNSConfig

    def _get_config_key(self) -> str:
        """Return config key for ALNS."""
        return "alns"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Run ALNS solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        routes, profit, solver_cost = run_alns(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            values,
            mandatory_nodes,
        )
        return routes, profit, solver_cost
