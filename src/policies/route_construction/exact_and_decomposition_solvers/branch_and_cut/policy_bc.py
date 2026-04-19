"""
Branch-and-Cut (BC) Policy Adapter.

Adapts the core Branch-and-Cut solver logic to the systems-agnostic
policy interface, handling parameter mapping, profit calculation, and environment
integration.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import BCConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .bc import BranchAndCutSolver
from .params import BCParams


@GlobalRegistry.register(
    PolicyTag.EXACT,
    PolicyTag.MATH_PROGRAMMING,
    PolicyTag.SOLVER,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("bc")
class BranchAndCutPolicy(BaseRoutingPolicy):
    """
    Adapter for the Branch-and-Cut routing solver.

    This policy implements an exact optimization method that combines:
    - Cutting planes to strengthen LP relaxations
    - Branch-and-bound for integer optimization
    - Separation algorithms for subtour elimination and capacity constraints

    As an exact solver, it provides provable optimality guarantees within
    the specified MIP gap and time limit.
    """

    def __init__(self, config: Optional[Union[BCConfig, Dict[str, Any]]] = None):
        """Initialize the BC policy adapter.

        Args:
            config: A typed BCConfig dataclass, a raw dictionary for parsing,
                or None to use framework defaults.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration dataclass type for automatic parsing."""
        return BCConfig

    def _get_config_key(self) -> str:
        """Return the unique identification key for this policy's configuration."""
        return "bc"

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
        Execute the Branch-and-Cut (BC) solver logic.

        This method coordinates an exact optimization search that strengthens linear
        programming (LP) relaxations using dynamically generated valid
        inequalities (cuts). The process alternates between branching on fractional
        variables and separating cuts—such as Subtour Elimination Constraints
        (SECs) and rounded capacity inequalities—to tighten the lower bounds
        until global optimality is reached or the gap threshold is met.

        In this implementation:
        - Cutting Planes: SECs are separated using min-cut algorithms (e.g.,
          Gomory-Hu trees or max-flow) on the fractional support graph.
        - Branching: Standard integer branching on arc selection variables.
        - Warm Start: Heuristics are used to find initial feasible solutions for
          early pruning.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                BC parameters (mip_gap, time_limit, cuts_enabled).
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
        n_nodes = len(sub_dist_matrix)

        # Build VRPP model
        model = VRPPModel(
            n_nodes=n_nodes,
            cost_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            revenue_per_kg=revenue,
            cost_per_km=cost_unit,
            mandatory_nodes=set(mandatory_nodes),
        )

        # Create solver with standardized configuration parameters
        params = BCParams.from_config(values)
        if "profit_aware_operators" in kwargs:
            params.profit_aware_operators = kwargs["profit_aware_operators"]
        if "vrpp" in kwargs:
            params.vrpp = kwargs["vrpp"]

        solver = BranchAndCutSolver(
            model=model,
            params=params,
        )

        # Solve
        routes, profit, stats = solver.solve()

        # Compute travel cost from the solver's objective value.
        # Objective = travel_cost - waste_collected (Gurobi minimizes)
        # profit = -ObjVal = waste_collected - travel_cost
        # => travel_cost = waste_collected - profit

        # 1. Total waste collected (revenue side)
        total_revenue = sum(model.get_node_profit(i) for route in routes for i in route)

        # 2. Derive distance cost: dist_cost = travel_cost
        # travel_cost = waste_collected - profit (already in monetary units)
        dist_cost = max(0.0, total_revenue - profit)

        return routes, profit, dist_cost
