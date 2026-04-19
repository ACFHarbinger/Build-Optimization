"""
PSOMA Policy Adapter.

Adapts the Particle Swarm Optimization Memetic Algorithm (PSOMA) to the
agnostic BaseRoutingPolicy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.psoma import PSOMAConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import PSOMAParams
from .solver import PSOMAsSolver


@RouteConstructorRegistry.register("psoma")
class PSOMAPolicy(BaseRoutingPolicy):
    """
    PSOMA policy class.

    Visits bins using Particle Swarm Optimization with a memetic local-search step.
    """

    def __init__(self, config: Optional[Union[PSOMAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return PSOMAConfig

    def _get_config_key(self) -> str:
        return "psoma"

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
        Execute the Particle Swarm Optimization Memetic Algorithm (PSOMA)
        solver logic.

        PSOMA is a hybrid metaheuristic that combines the collective intelligence
        of Particle Swarm Optimization (PSO) with the individual refinement
        capabilities of a Memetic Algorithm. While the swarm particles explore
        the global profit surface through velocity-driven movements (cognitive
        and social attraction), individual particles are periodically refined
        using local search operators. This "education" step ensures that
        particles converge onto meaningful local optima, effectively bridging
        long-range exploration with rigorous intensification.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                PSOMA parameters (pop_size, omega, c1, c2, local_search_freq).
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
        params = PSOMAParams(
            pop_size=int(values.get("pop_size", 20)),
            omega=float(values.get("omega", 0.4)),
            c1=float(values.get("c1", 1.5)),
            c2=float(values.get("c2", 2.0)),
            max_iterations=int(values.get("max_iterations", 200)),
            local_search_freq=int(values.get("local_search_freq", 10)),
            n_removal=int(values.get("n_removal", 2)),
            local_search_iterations=int(values.get("local_search_iterations", 500)),
            time_limit=float(values.get("time_limit", 60.0)),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        solver = PSOMAsSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        routes, profit, cost = solver.solve()
        return routes, profit, cost
