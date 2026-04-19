from typing import Any, Dict, List, Optional, Tuple, Type

from logic.src.configs.policies.abpc_hg import ABPCHGConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .scenario_prize_engine import ScenarioPrizeEngine
from .temporal_benders import TemporalBendersCoordinator


@GlobalRegistry.register(
    PolicyTag.MATHEURISTIC,
    PolicyTag.DECOMPOSITION,
    PolicyTag.REINFORCEMENT_LEARNING,
    PolicyTag.MULTI_PERIOD,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("abpc_hg")
class ABPCHGPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Adaptive Branch-and-Price-and-Cut with Heuristic Guidance (ABPC-HG).

    This policy wraps the exact BPC components but injects advanced matheuristic
    and scenario-based logic to solve Stochastic Multi-Period VRPs.
    """

    def __init__(self, config: Optional[ABPCHGConfig] = None):
        super().__init__(config)
        self.gamma = getattr(self.config, "gamma", 0.95) if self.config else 0.95

    @classmethod
    def _config_class(cls) -> Type[ABPCHGConfig]:
        return ABPCHGConfig

    def _get_config_key(self) -> str:
        return "abpc_hg"

    def _run_multi_period_solver(
        self,
        problem: ProblemContext,
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """
        Execute the ABPC-HG pipeline for the multi-day horizon.

        Args:
            problem: Current ProblemContext containing state data.
            multi_day_ctx: Optional context for spanning multiple days.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
                - today_solution: Standardized solution context for Day 0.
                - full_plan: Collection plan spanning the entire horizon.
                - stats: Execution statistics and Benders iteration metadata.
        """
        tree = problem.scenario_tree
        if tree is None:
            raise ValueError("ABPC-HG requires a ScenarioTree in ProblemContext.")

        capacity = problem.capacity
        revenue = problem.revenue_per_kg
        cost_unit = problem.cost_per_km
        # Step 1: Initialize the ScenarioTree (already passed in via context)

        # Step 2: Compute scenario-augmented node prizes (managed by the engine)
        prize_engine = ScenarioPrizeEngine(
            scenario_tree=tree,
            gamma=self.gamma,
            tau=capacity,
        )

        # Step 3 & 4: Execute Temporal Benders Master Problem
        coordinator = TemporalBendersCoordinator(
            tree=tree,
            prize_engine=prize_engine,
            capacity=capacity,
            revenue=revenue,
            cost_unit=cost_unit,
        )

        # The coordinator manages the Benders iterations, internally using:
        # - ProgressiveHedgingCGLoop at the root node.
        # - ALNSMultiPeriodPricer for heuristic column generation.
        # - MLBranchingStrategy and ScenarioConsistentBranching.
        # - DiveAndPricePrimalHeuristic for upper bounds.
        # - FixAndOptimizeRefiner for polishing.

        raw_plan, total_expected_profit = coordinator.solve(**problem.extra)

        today_route = raw_plan[0][0] if raw_plan and raw_plan[0] else []
        sol = SolutionContext.from_problem(problem, today_route)

        return (
            sol,
            raw_plan,
            {
                "policy": "abpc_hg",
                "expected_profit": total_expected_profit,
            },
        )
