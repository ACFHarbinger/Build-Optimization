"""
Hyper-ACO Policy Adapter.

Adapts the Hyper-Heuristic ACO solver to the common policy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import ACOConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.ant_colony_optimization.hyper_heuristic_aco import run_hyper_heuristic_aco

from .factory import PolicyRegistry


@PolicyRegistry.register("hyper_aco")
class HyperACOPolicy(BaseRoutingPolicy):
    """
    Hyper-Heuristic ACO policy class.

    Uses ACO to construct sequences of local search operators.
    """

    def __init__(self, config: Optional[Union[ACOConfig, Dict[str, Any]]] = None):
        """Initialize Hyper-ACO policy with optional config.

        Args:
            config: ACOConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return ACOConfig

    def _get_config_key(self) -> str:
        """Return config key for Hyper-ACO."""
        return "hyper_aco"

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
        Run Hyper-ACO solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        routes, profit, solver_cost = run_hyper_heuristic_aco(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            values,
            mandatory_nodes=mandatory_nodes,
        )

        return routes, profit, solver_cost
