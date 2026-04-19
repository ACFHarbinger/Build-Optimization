"""
BPC Policy Adapter.

Adapts the Branch-and-Price-and-Cut (BPC) logic to the agnostic interface.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import BPCConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine import (
    run_bpc,
)

from .params import BPCParams


@GlobalRegistry.register(
    PolicyTag.EXACT,
    PolicyTag.DECOMPOSITION,
    PolicyTag.MATH_PROGRAMMING,
    PolicyTag.SOLVER,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("bpc")
class BPCPolicy(BaseRoutingPolicy):
    """
    Branch-and-Price-and-Cut policy class.

    Visits pre-selected 'mandatory' bins using exact or heuristic BPC solvers.
    """

    def __init__(self, config: Optional[Union[BPCConfig, Dict[str, Any]]] = None):
        """Initialize BPC policy with optional config.

        Args:
            config: BPCConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return BPCConfig

    def _get_config_key(self) -> str:
        """Return config key for BPC."""
        return "bpc"

    def execute(
        self, **kwargs: Any
    ) -> Tuple[Union[List[int], List[List[int]]], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
        """
        Execute the Branch-and-Price-and-Cut (BPC) solver logic.

        This method coordinates the execution of the BPC algorithm, which is the
        most advanced exact optimization technique in the framework for single-day
        deterministic routing.

        Args:
            **kwargs: Context dictionary containing parameters needed for the subproblem.

        Returns:
            Tuple[Union[List[int], List[List[int]]], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
                A 5-tuple containing:
                - tour: The optimized collection routes (flat or nested).
                - cost: Total travel cost of the routes.
                - profit: Total net profit (Revenue + Future Value - Cost).
                - search_context: The enriched search context after BPC execution.
                - multi_day_context: The final multi-day state metadata.
        """
        config_dict = kwargs.get("config", {}).get(self._get_config_key(), {})

        # Unpack problem data from kwargs
        dist_matrix = kwargs["model_ls"][1]
        bins = kwargs["bins"]
        profit_vars = kwargs["model_ls"][2]

        R = profit_vars.get("revenue_kg", 1.0)
        C = profit_vars.get("cost_km", 1.0)
        vehicle_limit = profit_vars.get("number_vehicles", None)

        wastes = {i: float(bins.c[i - 1]) for i in range(1, len(bins.c) + 1)}
        capacity = float(profit_vars.get("bin_capacity", 100.0))
        mandatory = set(kwargs.get("mandatory", []))

        params = BPCParams.from_config(config_dict)

        n_bins = len(bins.c)
        node_prizes: Dict[int, float] = {}

        # Pure Single-Day Mode (base BPC)
        for i in range(1, n_bins + 1):
            node_prizes[i] = wastes[i] * R

        routes, profit = run_bpc(
            dist_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            params=params,
            mandatory_indices=mandatory,
            vehicle_limit=vehicle_limit,
            node_prizes=node_prizes,
        )

        global_route = []
        if routes:
            for r in routes:
                global_route.extend([n for n in r if n != 0])
                global_route.append(0)

        model_env = kwargs.get("model_env")
        cost = model_env.compute_route_cost(global_route) if model_env is not None else 0.0

        return global_route, cost, profit, kwargs.get("search_context"), kwargs.get("multi_day_context")

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
        Execute core BPC optimization combining column generation and cutting planes.

        This method coordinates the highest-fidelity exact search in the
        framework. It dynamically generates routes (columns) with positive
        reduced cost and strengthens the master problem relaxation using valid
        inequalities (cuts) like SECs and Rounded Capacity Inequalities. The
        search is managed within a global branch-and-bound tree.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                BPC parameters (branching_strategy, cuts_enabled, time_limit).
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
        # Return contract for run_bpc:
        #   routes          — list of customer-node lists (depot excluded)
        #   objective_value — net profit = Σ(revenue_i) - travel_cost, in monetary units.
        #                     May be a greedy-fallback value if BPC found no integer solution.
        # Convert local mandatory indices to a set of mandatory nodes for the solver
        mandatory_indices: Set[int] = set(mandatory_nodes)

        # Initialize standardized params object (Phase 1 refactoring)
        params = BPCParams.from_config(values)

        # Extract vehicle limit from simulation context (sim.n_vehicles)
        n_vehicles = kwargs.get("n_vehicles")
        # Explicit int conversion and positive check. None and 0 both map to
        # unlimited fleet. False is rejected at the int() call (TypeError surfaced
        # to the caller) rather than silently treated as unlimited.
        vehicle_limit = None if n_vehicles is None else int(n_vehicles) if int(n_vehicles) > 0 else None

        # run_bpc returns (routes, objective_value) where objective_value is
        # net profit (revenue - travel_cost) in the problem's monetary units.
        # It is NOT a raw travel cost despite the variable name used in run_bpc's
        # return signature. Rename immediately to prevent future misreading.
        routes, objective_value = run_bpc(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_indices=mandatory_indices,
            vehicle_limit=vehicle_limit,
            env=kwargs.get("model_env"),
            node_coords=kwargs.get("node_coords"),
            recorder=kwargs.get("recorder"),
        )

        profit = objective_value

        # Compute raw travel distance (km)
        raw_distance = 0.0
        for route in routes:
            # Normalize: strip any leading/trailing depot index before wrapping.
            # Route.nodes stores customer-only sequences, but defensive stripping
            # guards against representation changes in run_bpc's return value.
            inner = [n for n in route if n != 0]
            if not inner:
                continue

            path = [0] + inner + [0]
            for i in range(len(path) - 1):
                raw_distance += sub_dist_matrix[path[i]][path[i + 1]]

        return routes, profit, raw_distance
