"""
ACO Policy Adapter.

Adapts the K-Sparse Ant Colony Optimization solver to the common policy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import ACOConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.ant_colony_optimization.k_sparse_aco import run_k_sparse_aco

from .factory import PolicyRegistry


@PolicyRegistry.register("aco")
class ACOPolicy(BaseRoutingPolicy):
    """
    K-Sparse Ant Colony Optimization policy class.

    Uses ACS with sparse pheromone matrix for efficient VRP solving.
    """

    def __init__(self, config: Optional[Union[ACOConfig, Dict[str, Any]]] = None):
        """Initialize ACO policy with optional config.

        Args:
            config: ACOConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return ACOConfig

    def _get_config_key(self) -> str:
        """Return config key for ACO."""
        return "aco"

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
        Run K-Sparse ACO solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        routes, profit, solver_cost = run_k_sparse_aco(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            values,
            mandatory_nodes=mandatory_nodes,
        )
        return routes, profit, solver_cost
