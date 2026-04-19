r"""
Scenario Tree Extensive Form (ST-EF) Policy Adapter for Stochastic VRPP.

ST-EF solves the Multi-Period Stochastic Integer Routing Problem by constructing
the Deterministic Equivalent Problem (DEP) over the entire `ScenarioTree`.

Mathematical Principle:
    Given a scenario tree $\mathcal{T}$ with nodes $n \in \mathcal{N}$, the model
    minimizes the expected cost across all leaf nodes while enforcing
    non-anticipativity implicitly through the tree structure (since branches share
    ancestry).

    The formulation solves:
        max  Σ_{n \in \mathcal{N}} p_n [ Revenue(n) - TravelCost(n) - Penalty(n) ]
        s.t. Vehicle capacity at each node n
             Flow conservation across days (Day t -> Day t+1) based on branching
             Bin fill-level transitions w_i(n) = w_i(parent(n)) + increment_i(n)

Algorithm — Integrated Rolling Horizon:
    1.  **Lookahead**: At each simulation day $d$, ST-EF "sees" $T$ days into the
        future through the generated `ScenarioTree`.
    2.  **State Mapping**: Maps current simulation bin inventories to the
        tree's root node.
    3.  **Global Optimization**: Solves a single monolithic MILP (via Gurobi)
        covering all scenarios in the tree simultaneously.
    4.  **Action Selection**: Extracts only the Day 0 decision (the current day's
        routing plan) and implements it, discarding future-day decisions which
        will be re-evaluated on the next rolling horizon step.

Complexity:
    The size of the EF model grows exponentially with the branching factor and
    horizon depth $T$. It provides the theoretical upper bound on solution
    quality (Perfect Information or SAA-optimal) but requires significant
    computational resources.

Registry key: ``"st_ef"``
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.configs.policies.st_ef import ScenarioTreeExtensiveFormConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine import (
    ScenarioTreeExtensiveFormEngine,
)


@GlobalRegistry.register(
    PolicyTag.EXACT,
    PolicyTag.STOCHASTIC,
    PolicyTag.MULTI_PERIOD,
    PolicyTag.SOLVER,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("st_ef")
class ScenarioTreeExtensiveFormPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Adapter for the Scenario-Tree Extensive Form (ST-EF) policy.
    Now standardized to the Multi-Period framework.
    """

    @classmethod
    def _config_class(cls):
        return ScenarioTreeExtensiveFormConfig

    def _get_config_key(self) -> str:
        """Return the configuration key."""
        return "st_ef"

    def _run_multi_period_solver(
        self,
        problem: ProblemContext,
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """
        Execute the Scenario Tree Extensive Form (ST-EF) solver logic.

        This method solves the Deterministic Equivalent Problem (DEP) of the
        multi-period stochastic routing problem. It constructs a single large-scale
        Mixed-Integer Linear Programming (MILP) model that encompasses all bin
        fill realization scenarios in the tree simultaneously. Non-anticipativity
        constraints are implicitly satisfied by the tree's branching structure,
        ensuring Day 0 decisions are optimal with respect to the expected
        cumulative profit across the entire horizon.

        Args:
            problem: The current ProblemContext containing all state data.
            multi_day_ctx: Optional context for spanning multiple days.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
                - today_solution: Standardized solution context for Day 0.
                - full_plan: Collection plan (Day 0 routes specifically).
                - stats: Execution statistics and solver performance metadata.
        """
        tree = problem.scenario_tree
        if tree is None:
            raise ValueError("ST-EF requires a ScenarioTree in ProblemContext.")

        # The ST-EF engine needs adaptation to the new ScenarioTree
        engine = ScenarioTreeExtensiveFormEngine(
            tree=tree,  # type: ignore[arg-type]
            distance_matrix=problem.distance_matrix,
            wastes=problem.wastes,
            capacity=problem.capacity,
            waste_weight=getattr(self.config, "waste_weight", 1.0),
            cost_weight=getattr(self.config, "cost_weight", 1.0),
            overflow_penalty=getattr(self.config, "overflow_penalty", 500.0),
            time_limit=getattr(self.config, "time_limit", 300.0),
        )

        # ST-EF traditionally returns the Day 0 route specifically
        route, expected_val = engine.solve()

        # Wrap plan into [day][vehicle][node]
        full_plan: List[List[List[int]]] = [[] for _ in range(self.horizon + 1)]
        full_plan[0] = [route]

        sol_ctx = SolutionContext.from_problem(problem, route)

        return sol_ctx, full_plan, {"mip_status": "solved", "expected_profit": expected_val}

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """Legacy fallback."""
        return [], 0.0, 0.0
