"""
Base Multi-Period Routing Policy.

Provides a template for routing policies that perform T-period optimization
over a scenario tree, rather than myopic single-day routing.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.pipeline.simulations.bins.prediction import ScenarioTree

if TYPE_CHECKING:
    pass

from .base_routing_policy import BaseRoutingPolicy


class BaseMultiPeriodRoutingPolicy(BaseRoutingPolicy):
    """
    Base class for policies optimized for stochastic multi-period routing (SIRP).

    This class provides utilities for:
    - Horizon management (T days).
    - Scenario tree traversal.
    - Future mandatory node prediction.
    - Overflow (stockout) penalty calculation.
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.horizon: int = getattr(self.config, "horizon", 7)
        self.stockout_penalty: float = getattr(self.config, "stockout_penalty", 500.0)

    def _get_scenario_tree(self, kwargs: Dict[str, Any]) -> ScenarioTree:
        """Extract the scenario tree from the simulation context."""
        tree = kwargs.get("scenario_tree")
        if tree is None:
            raise ValueError(f"{type(self).__name__} requires a ScenarioTree in the context.")
        return tree

    def _predict_mandatory_nodes_for_horizon(
        self, tree: ScenarioTree, initial_mandatory: List[int]
    ) -> Dict[int, List[int]]:
        """
        Predict mandatory nodes for each day in the horizon based on the tree.

        Returns:
            Dict[day, list_of_bin_ids]
        """
        mandatory_map: Dict[int, List[int]] = {0: initial_mandatory}

        # We traverse the main scenario path (or all branches) to identify
        # when bins are expected to overflow (>100%).
        for t in range(1, tree.horizon + 1):
            scenarios = tree.get_scenarios_at_day(t)
            # For simplicity, we mark a node as mandatory if it overflows in ANY scenario at day t,
            # but hasn't been collected in previous days t' < t.
            # Real solvers might handle this more robustly.
            day_mandatory = []
            for scenario in scenarios:
                overflow_bins = np.where(scenario.wastes >= 100.0)[0] + 1  # 1-based
                for b_id in overflow_bins:
                    if b_id not in day_mandatory:
                        # Check if it was already mandatory in a previous day
                        already_seen = False
                        for prev_t in range(t):
                            if b_id in mandatory_map.get(prev_t, []):
                                already_seen = True
                                break
                        if not already_seen:
                            day_mandatory.append(int(b_id))

            mandatory_map[t] = day_mandatory

        return mandatory_map

    def _calculate_stockout_costs(self, inventory_sequence: np.ndarray) -> float:
        """
        Calculate penalties for waste overflows (>100%).

        Args:
            inventory_sequence: array of shape (N, T) with fill levels.

        Returns:
            Total penalty cost.
        """
        overflows = np.maximum(0, inventory_sequence - 100.0)
        return float(np.sum(overflows) * self.stockout_penalty)

    @abstractmethod
    def _run_multi_period_solver(
        self,
        problem: "ProblemContext",
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple["SolutionContext", List[List[List[int]]], Dict[str, Any]]:
        """
        Solve the T-period optimization problem.

        Args:
            problem:       Fully typed problem descriptor for today.
            multi_day_ctx: Current multi-day state (may be None on day 0).

        Returns:
            Tuple of:
                solution   – SolutionContext for today's routes only.
                full_plan  – List[day][vehicle][node] for the whole horizon
                             (to be stored in MultiDayContext.full_plan_snapshot).
                metadata   – Solver telemetry dict.
        """
        pass

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
        Dummy implementation of the single-period solver.

        Multi-period policies override the execute() method directly (or rely on
        the BaseMultiPeriodRoutingPolicy override) to call _run_multi_period_solver,
        so this method is never invoked in the SIRP workflow.
        """
        return [], 0.0, 0.0

    def execute(
        self, **kwargs: Any
    ) -> Tuple[Union[List[int], List[List[int]]], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
        from logic.src.interfaces.context.problem_context import ProblemContext

        # 1. Load area params (unchanged — uses existing _load_area_params)
        capacity, revenue, cost_unit, values = self._load_area_params(
            kwargs.get("area", "Rio Maior"),
            kwargs.get("waste_type", "plastic"),
            kwargs.get("config", {}),
        )

        # 2. Build typed problem descriptor
        problem = ProblemContext.from_kwargs(kwargs, capacity, revenue, cost_unit)

        multi_day_ctx: Optional[MultiDayContext] = kwargs.get("multi_day_context")
        search_ctx: Optional[SearchContext] = kwargs.get("search_context")

        # 3. Solve
        solution, full_plan, solver_metadata = self._run_multi_period_solver(problem, multi_day_ctx)

        # 4. Update MultiDayContext with full plan snapshot
        if multi_day_ctx is not None:
            multi_day_ctx = multi_day_ctx.update(
                full_plan_snapshot=full_plan,
                extra={**multi_day_ctx.extra, **solver_metadata},
            )

        # 5. Return in the existing 5-tuple format
        tour = solution.to_flat_tour()
        cost = solution.total_cost
        profit = solution.total_profit

        return tour, cost, profit, search_ctx, multi_day_ctx
