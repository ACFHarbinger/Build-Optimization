"""
ILS-RVND-SP Policy Adapter.

Adapts the Iterated Local Search - Randomized Variable Neighborhood Descent - Set Partitioning (ILS-RVND-SP) logic to the agnostic interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import ILSRVNDSPConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp import (
    ILSRVNDSPSolver,
)
from logic.src.policies.route_construction.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.params import (
    ILSRVNDSPParams,
)


@GlobalRegistry.register(
    PolicyTag.MATHEURISTIC,
    PolicyTag.LOCAL_SEARCH,
    PolicyTag.DECOMPOSITION,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("ils_rvnd_sp")
class ILSRVNDSPPolicy(BaseRoutingPolicy):
    """
    ILS-RVND-SP policy class.

    Visits pre-selected 'mandatory' bins using Iterated Local Search, Randomized Variable Neighborhood Descent, and Set Partitioning.
    """

    def __init__(self, config: Optional[Union[ILSRVNDSPConfig, Dict[str, Any]]] = None):
        """Initialize ILS-RVND-SP policy with optional config.

        Args:
            config: ILSRVNDSPConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return ILSRVNDSPConfig

    def _get_config_key(self) -> str:
        """Return config key for ILS-RVND-SP."""
        return "ils_rvnd_sp"

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
        Execute the Iterated Local Search - Randomized Variable Neighborhood Descent -
        Set Partitioning (ILS-RVND-SP) solver logic.

        ILS-RVND-SP is a powerful matheuristic that operates in two main stages:
        1. Heuristic Generation (ILS + RVND): Uses a randomized local search
           framework to discover a diverse set of feasible routes by iteratively
           perturbing and refining solution structures.
        2. Set Partitioning (SP) Optimization: All discovered routes are collected
           into a pool. A Set Partitioning MIP is then solved (using Gurobi) to
           select the optimal subset of routes that covers the required nodes
           while maximizing profit.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                ILS settings, RVND neighborhood configurations, and SP MIP goals.
            mandatory_nodes (List[int]): Local indices of bins that MUST be
                collected in this period.
            **kwargs: Additional context, including:
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple[List[List[int]], float, float]: A 3-tuple containing:
                - routes: Optimized collection routes for the current day.
                - profit: Total calculated net profit (Total Revenue - Total Cost).
                - cost: Total travel cost calculated by the solver.
        """
        params = ILSRVNDSPParams(
            max_restarts=int(values.get("max_restarts", 10)),
            max_iter_ils=int(values.get("max_iter_ils", 100)),
            perturbation_strength=int(values.get("perturbation_strength", 2)),
            use_set_partitioning=bool(values.get("use_set_partitioning", True)),
            mip_time_limit=float(values.get("mip_time_limit", 60.0)),
            sp_mip_gap=float(values.get("sp_mip_gap", 0.01)),
            N=int(values.get("N", 150)),
            A=float(values.get("A", 11.0)),
            MaxIter_a=int(values.get("MaxIter_a", 50)),
            MaxIter_b=int(values.get("MaxIter_b", 100)),
            MaxIterILS_b=int(values.get("MaxIterILS_b", 2000)),
            TDev_a=float(values.get("TDev_a", 0.05)),
            TDev_b=float(values.get("TDev_b", 0.005)),
            max_vehicles=int(values.get("max_vehicles", 9999)),
            time_limit=float(values.get("time_limit", 300.0)),
            seed=int(values.get("seed", 42)),
            vrpp=bool(values.get("vrpp", True)),
            profit_aware_operators=bool(values.get("profit_aware_operators", False)),
            local_search_iterations=int(values.get("local_search_iterations", 500)),
        )

        solver = ILSRVNDSPSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        routes, best_profit, best_cost = solver.solve()

        return routes, best_profit, best_cost
