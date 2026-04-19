"""
Memetic Algorithm with Island Model (MA-IM) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ma_im import MemeticAlgorithmIslandModelConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import MemeticAlgorithmIslandModelParams
from .solver import MemeticAlgorithmIslandModelSolver


@RouteConstructorRegistry.register("ma_im")
class MemeticAlgorithmIslandModelPolicy(BaseRoutingPolicy):
    """
    Memetic Algorithm Island Model policy class.
    """

    def __init__(self, config: Optional[Union[MemeticAlgorithmIslandModelConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return MemeticAlgorithmIslandModelConfig

    def _get_config_key(self) -> str:
        return "ma_im"

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
        Execute the Memetic Algorithm Island Model (MA-IM) solver logic.

        MA-IM is a parallelized memetic architecture that partitions the
        population into multiple independent "islands". Each island runs its own
        evolutionary and local search cycle. Periodically, individuals migrate
        between islands (information exchange) to maintain global diversity
        while allowing each island to exploit its local search space effectively.
        This prevents the "super-individual" problem where a single high-quality
        solution dominates the entire population too early.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                MA-IM parameters (n_islands, island_size, max_generations).
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
        params = MemeticAlgorithmIslandModelParams(
            n_islands=values.get("n_islands", 5),
            island_size=values.get("island_size", 4),
            max_generations=values.get("max_generations", 50),
            stagnation_limit=values.get("stagnation_limit", 5),
            n_removal=values.get("n_removal", 1),
            local_search_iterations=values.get("local_search_iterations", 100),
            time_limit=values.get("time_limit", 60.0),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        solver = MemeticAlgorithmIslandModelSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        routes, profit, cost = solver.solve()
        return routes, profit, cost
