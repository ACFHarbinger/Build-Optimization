from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.configs.policies.lbbd import LBBDConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

if TYPE_CHECKING:
    pass

from .lbbd_engine import LBBDEngine


@GlobalRegistry.register(
    PolicyTag.EXACT,
    PolicyTag.DECOMPOSITION,
    PolicyTag.MULTI_PERIOD,
)
@RouteConstructorRegistry.register("lbbd")
class LBBDPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Adapter for the Logic-Based Benders Decomposition (LBBD) policy.
    Supports multi-period stochastic decision making via ScenarioTree.
    """

    @classmethod
    def _config_class(cls):
        return LBBDConfig

    def _get_config_key(cls) -> str:
        return "lbbd"

    def _run_multi_period_solver(
        self,
        problem: ProblemContext,
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """
        Execute the Logic-Based Benders Decomposition (LBBD) solver logic.

        This method decomposes the multi-period stochastic routing problem into
        a Master Problem (scheduling assignment) and several Sub-problems
        (specific daily routing). It utilizes logic-based cuts to communicate
        between the scheduling level and the routing level, ensuring that
        Day 0 routes are optimized with respect to both immediate profit and
        expected longer-term resource availability across scenarios.

        Args:
            problem: The current ProblemContext containing all state data.
            multi_day_ctx: Optional context for spanning multiple days.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
                - today_solution: Standardized solution context for Day 0.
                - full_plan: Collection plan (nested list by day and vehicle).
                - stats: Execution statistics and decomposition iterations.
        """
        tree = problem.scenario_tree
        if tree is None:
            raise ValueError("LBBD requires a ScenarioTree in ProblemContext.")

        distance_matrix = problem.distance_matrix
        cfg: LBBDConfig = self.config

        # Instantiate and run Engine
        engine = LBBDEngine(
            config=cfg,
            distance_matrix=distance_matrix,
            tree=tree,
            capacity=problem.capacity,
        )

        # Solve for the full horizon T
        full_plan, profit, stats = engine.solve()

        # Extract Day 0 route
        today_route = full_plan[0][0] if full_plan and full_plan[0] else []
        sol_ctx = SolutionContext.from_problem(problem, today_route)

        return sol_ctx, full_plan, stats

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
        Satisfies the abstract base class, but LBBD uses _run_multi_period_solver.
        """
        return [], 0.0, 0.0
