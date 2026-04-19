import copy
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.hyper_heuristics.selection_hyper_heuristic.params import SHHParams
from logic.src.utils.policy.llh_pool import LLHPool
from logic.src.utils.policy.routes import (
    route_profit,
)
from logic.src.utils.policy.wrappers import (
    greedy_day_route,
)


@GlobalRegistry.register(
    PolicyTag.HYPER_HEURISTIC,
    PolicyTag.ORCHESTRATOR,
    PolicyTag.MULTI_PERIOD,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("shh")
class SelectionHHPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Selection Hyper-Heuristic (SHH).
    Controls a set of Low-Level Heuristics (LLHs).
    Uses a reinforcement learning-like mechanism (UCB or simple probability updates)
    to select the next LLH to apply, then accepts or rejects using Late Acceptance.

    Registry key: ``"shh"``
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.params = SHHParams.from_config(config)
        self.iters = self.params.iters
        self.history_len = self.params.history_len
        self.seed = self.params.seed
        self.rng = random.Random(self.seed)
        self.llhs = LLHPool.get_all()

    def _evaluate(self, plan: List[List[List[int]]], problem: ProblemContext) -> float:
        tot = 0.0
        cur_prob = problem
        for d in range(problem.horizon):
            rt = plan[d][0] if plan[d] else []
            tot += route_profit(rt, cur_prob)
            cur_prob = cur_prob.advance(rt)
        return tot

    def _run_multi_period_solver(
        self,
        problem: ProblemContext,
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """
        Execute the Selection Hyper-Heuristic (SHH) solver logic.

        SHH is a high-level search methodology that operates on a search space
        of heuristics (Low-Level Heuristics, or LLHs) rather than directly
        modifying the routing solution. It manages a pool of operators and
        learns which LLHs are most effective for the current problem state.

        Search Logic:
        1. **Selection Mechanism**: Uses a reinforcement learning-like approach
           (probability matching) to choose an LLH based on its historical
           performance.
        2. **Execution**: Applies the selected LLH (e.g., ruin-and-recreate, 2-opt)
           to the current multi-period collection plan.
        3. **Acceptance Criterion**: Employs Late Acceptance Hill Climbing (LAHC)
           to decide whether to accept the new plan, preventing the search from
           getting trapped in local optima while maintaining efficiency.

        In this implementation, SLH optimizes Day 0 and future days through
        collaborative operator guidance.

        Args:
            problem: The current ProblemContext containing state data.
            multi_day_ctx: Optional context for spanning multiple rolling days.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
                - today_solution: Standardized solution context for Day 0.
                - best_plan: Collection plan (nested list by day and vehicle).
                - stats: Execution statistics (LLH usage, iterations, etc.).
        """
        np_rng = np.random.default_rng(self.seed)

        cur_plan = []
        cur_prob = problem
        for _d in range(problem.horizon):
            rt = greedy_day_route(cur_prob, np_rng)
            cur_plan.append([rt])
            cur_prob = cur_prob.advance(rt)

        best_plan = copy.deepcopy(cur_plan)
        best_prof = self._evaluate(best_plan, problem)

        # Late Acceptance history
        la_history = [best_prof] * self.history_len

        # LLH weights
        weights = [1.0] * len(self.llhs)

        cur_prof = best_prof

        for k in range(self.iters):
            # Selection
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            idx = self.rng.choices(range(len(self.llhs)), weights=probs, k=1)[0]
            llh = self.llhs[idx]

            # Apply
            new_plan = llh(cur_plan, problem, self.rng)
            new_prof = self._evaluate(new_plan, problem)

            # Acceptance (Late Acceptance Hill Climbing)
            v = k % self.history_len
            if new_prof >= la_history[v] or new_prof >= cur_prof:
                cur_plan = copy.deepcopy(new_plan)
                cur_prof = new_prof

                # Reward LLH
                weights[idx] += 1.0

                if new_prof > best_prof:
                    best_prof = new_prof
                    best_plan = copy.deepcopy(new_plan)
            else:
                # Penalize slightly
                weights[idx] = max(0.1, weights[idx] - 0.1)

            la_history[v] = cur_prof

        today_route = best_plan[0][0] if best_plan[0] else []
        sol = SolutionContext.from_problem(problem, today_route)

        return sol, best_plan, {"shh_iters": self.iters, "expected_profit": best_prof}
