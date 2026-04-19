"""
Memetic Algorithm with Dual Population (MA-DP) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ma_dp import MemeticAlgorithmDualPopulationConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import MemeticAlgorithmDualPopulationParams
from .solver import MemeticAlgorithmDualPopulationSolver


@RouteConstructorRegistry.register("ma_dp")
class MemeticAlgorithmDualPopulationPolicy(BaseRoutingPolicy):
    """
    Memetic Algorithm Dual Population policy class.
    """

    def __init__(self, config: Optional[Union[MemeticAlgorithmDualPopulationConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return MemeticAlgorithmDualPopulationConfig

    def _get_config_key(self) -> str:
        return "ma_dp"

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
        Execute the Memetic Algorithm with Dual Population (MA-DP) solver logic.

        MA-DP is an specialized memetic architecture that maintains two distinct
        populations:
        - Diverse Population: Focuses on exploration and maintaining solution
          structural variety to prevent premature convergence.
        - Elite Population: Focuses on intensification and refining the best
          known solutions.
        Information is periodically exchanged between populations (diversity
        injection), and elite individuals undergo rigorous learning (local search)
        to ensure high-quality solution boundaries.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                MA-DP parameters (population_size, diversity_injection_rate, elite_count).
            mandatory_nodes (List[int]): Local indices of bins that MUST be
                collected in this period.
            **kwargs: Additional context, including:
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple[List[List[int]], float, float]: A 3-tuple containing:
                - routes: Optimized collection routes (list-of-lists, local indices).
                - profit: Total calculated net profit (Total Revenue - Total Cost).
                - cost: Total travel cost calculated by the solver.
        """
        params = MemeticAlgorithmDualPopulationParams(
            population_size=values.get("population_size", 30),
            max_iterations=values.get("max_iterations", 200),
            diversity_injection_rate=values.get("diversity_injection_rate", 0.2),
            elite_learning_weights=values.get("elite_learning_weights"),
            elite_count=values.get("elite_count", 3),
            local_search_iterations=values.get("local_search_iterations", 500),
            time_limit=values.get("time_limit", 300.0),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        solver = MemeticAlgorithmDualPopulationSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()
