"""
Branch-and-Price (BP) Policy Adapter.

Adapts the core Branch-and-Price solver logic to the systems-agnostic policy
interface, handling parameter mapping, profit calculation, and environment
integration.

References:
    Barnhart, C., Johnson, E. L., Nemhauser, G. L., Savelsbergh, M. W. P.,
    & Vance, P. H. (1998). "Branch-and-price: Column Generation for Solving
    Huge Integer Programs". Operations Research, 46(3), 316-329.

    Baldacci, R., Mingozzi, A., & Roberti, R. (2011). "New Route Relaxation
    and Pricing Strategies for the Vehicle Routing Problem". Operations
    Research, 59(5), 1269-1283.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import BPConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .bp import BranchAndPriceSolver
from .params import BPParams


@GlobalRegistry.register(
    PolicyTag.EXACT,
    PolicyTag.DECOMPOSITION,
    PolicyTag.MATH_PROGRAMMING,
    PolicyTag.SOLVER,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("bp")
class BranchAndPricePolicy(BaseRoutingPolicy):
    """
    Adapter for the Branch-and-Price routing solver.

    Implements a column generation method that handles large-scale optimisation
    by implicitly enumerating exponentially many variables (routes).

    The algorithm uses:

    1. Set Covering master problem — selects routes to cover all mandatory
       nodes while maximising profit.
    2. Pricing subproblem — generates profitable routes via RCSPP (exact DP
       or greedy heuristic).
    3. Column generation — iteratively adds routes with positive reduced cost.
    4. Branch-and-bound — optionally enforces integrality via edge branching
       or Ryan-Foster branching.

    Configuration
    -------------
    ``branching_strategy``  (``"edge"`` | ``"ryan_foster"``, default ``"edge"``)
        Selects the B&B branching scheme.

    ``use_exact_pricing``  (bool, default False)
        When True, uses the DP-based ``RCSPPSolver``; otherwise uses the
        greedy heuristic ``PricingSubproblem``.

    ``use_ng_routes``  (bool, default True)
        When ``use_exact_pricing=True``, enables the ng-route relaxation of
        Baldacci et al. (2011).  Dramatically reduces the label count on large
        instances while still producing near-elementary routes.  Set False to
        restore exact ESPPRC behaviour.

    ``ng_neighborhood_size``  (int, default 8)
        Size of each node's ng-neighborhood N_i (including the node itself).
        Larger values tighten the relaxation (approaching exact ESPPRC) at the
        cost of more labels per solve.
    """

    def __init__(self, config: Optional[Union[BPConfig, Dict[str, Any]]] = None) -> None:
        """
        Initialise the BP policy adapter.

        Args:
            config: A typed BPConfig dataclass, a raw dictionary for parsing,
                or None to use framework defaults.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration dataclass type for automatic parsing."""
        return BPConfig

    def _get_config_key(self) -> str:
        """Return the unique identification key for this policy's configuration."""
        return "bp"

    def execute(
        self, **kwargs: Any
    ) -> Tuple[Union[List[int], List[List[int]]], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
        """
        Execute the Branch-and-Price (BP) solver logic.

        This method coordinates the execution of the BP algorithm, which is an
        exact optimization technique for solving the VRPP using column
        generation within a branch-and-bound framework.

        If `multi_day_mode` is enabled (typically for stochastic sub-problems
        in a rolling horizon), it executes a specialized multi-period
        evaluation that bypasses standard subset extraction. Otherwise,
        it falls back to the standard policy execution loop.

        Args:
            **kwargs: Context dictionary containing:
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.
                - config (Dict): Optional nested configuration overrides.

        Returns:
            Tuple[Union[List[int], List[List[int]]], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
                A 5-tuple containing:
                - tour: The optimized collection routes.
                - cost: Total travel cost calculated based on the routes.
                - profit: Total calculated net profit (Total Revenue - Total Cost).
                - search_context: The enriched search context after column generation.
                - multi_day_context: The final multi-day state metadata.
        """
        config_dict = kwargs.get("config", {}).get(self._get_config_key(), {})
        multi_day_mode = config_dict.get("multi_day_mode", False)

        if not multi_day_mode:
            # Standard single-day mode
            return super().execute(**kwargs)

        # Multi-day mode (Exact SDP inner deterministic optimizer)
        dist_matrix = kwargs["model_ls"][1]
        bins = kwargs["bins"]
        profit_vars = kwargs["model_ls"][2]

        R = profit_vars.get("revenue_kg", 1.0)
        C = profit_vars.get("cost_km", 1.0)
        num_vehicles = profit_vars.get("n_vehicles", 0)

        wastes = {i: float(bins.c[i - 1]) for i in range(1, len(bins.c) + 1)}
        capacity = float(profit_vars.get("bin_capacity", 100.0))
        mandatory = kwargs.get("mandatory", [])

        params = BPParams.from_config(config_dict)

        solver = BranchAndPriceSolver(
            n_nodes=len(dist_matrix) - 1,
            cost_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            revenue_per_kg=R,
            cost_per_km=C,
            mandatory_nodes=set(mandatory),
            params=params,
            vehicle_limit=num_vehicles,
        )

        flat_tour, profit, _stats = solver.solve()

        # Preserve the exact structure provided by the solver
        global_route = flat_tour if flat_tour else [0, 0]

        # Explicitly check for None before invoking methods on Optional[Any]
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
        Execute core Branch-and-Price optimization using column generation.

        This method solves the master problem (set covering) by iteratively calling
        the pricing subproblem (RCSPP) to identify routes with positive reduced
        cost. The algorithm employs state-space relaxations (ng-routes) to
        maintain balance between lower-bound quality and computational
        complexity. If integrality is not achieved at the root, a branching
        scheme (Edge or Ryan-Foster) is invoked.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                BP parameters (ng_neighborhood_size, branching_strategy).
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
        n_nodes = len(sub_dist_matrix) - 1
        mandatory_set = set(mandatory_nodes)

        # ------------------------------------------------------------------
        # Build and run solver with standardized parameters
        # ------------------------------------------------------------------
        params = BPParams.from_config(values)

        solver = BranchAndPriceSolver(
            n_nodes=n_nodes,
            cost_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            revenue_per_kg=revenue,
            cost_per_km=cost_unit,
            mandatory_nodes=mandatory_set,
            params=params,
        )

        tour, profit, statistics = solver.solve()

        # ------------------------------------------------------------------
        # Convert flat tour to per-vehicle route lists (depot stripped)
        # ------------------------------------------------------------------
        if tour and len(tour) > 2:
            route = [node for node in tour if node != 0]
            routes = [route] if route else []
        else:
            routes = []

        # ------------------------------------------------------------------
        # Calculate actual distance cost
        # ------------------------------------------------------------------
        dist_cost = 0.0
        if tour:
            prev = 0
            for node in tour:
                if node != 0:
                    dist_cost += sub_dist_matrix[prev, node]
                    prev = node
                elif prev != 0:
                    dist_cost += sub_dist_matrix[prev, 0]
                    prev = 0
            dist_cost *= cost_unit

        return routes, profit, dist_cost
