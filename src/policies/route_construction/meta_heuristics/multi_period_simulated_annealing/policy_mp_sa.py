import copy
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.meta_heuristics.multi_period_simulated_annealing.params import MP_BMC_Params
from logic.src.utils.policy.routes import (
    route_profit,
)
from logic.src.utils.policy.wrappers import (
    greedy_day_route,
    two_opt,
)


@GlobalRegistry.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.TRAJECTORY_BASED,
    PolicyTag.LOCAL_SEARCH,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
    PolicyTag.MULTI_PERIOD,
)
@RouteConstructorRegistry.register("mp_sa")
class MultiPeriodSimulatedAnnealingPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Multi-Period Simulated Annealing metaheuristic.
    Accepts worse solutions with probability e^(delta/T).
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.params = MP_BMC_Params.from_config(config)
        self.max_iter = self.params.iters
        self.init_temp = self.params.init_temp
        self.cooling_rate = self.params.cooling_rate
        self.seed = self.params.seed
        self.rng = random.Random(self.seed)

    def _evaluate(self, plan: List[List[List[int]]], problem: ProblemContext) -> float:
        tot = 0.0
        cur_prob = problem
        for d in range(problem.horizon):
            rt = plan[d][0] if plan[d] else []
            tot += route_profit(rt, cur_prob)
            cur_prob = cur_prob.advance(rt)
        return tot

    def _neighbor(self, plan: List[List[List[int]]], problem: ProblemContext) -> List[List[List[int]]]:
        # swap 2 nodes between days, or add/drop
        new_plan = copy.deepcopy(plan)
        D = problem.horizon

        # We will just do a simple add/drop on a random day to avoid complex feasibility logic
        day = self.rng.randint(0, D - 1)
        rt = new_plan[day][0] if new_plan[day] else []

        if not rt or self.rng.random() < 0.5:
            # Add a node not in route
            cand = [v for v in range(1, len(problem.distance_matrix)) if v not in rt]
            if cand:
                v = self.rng.choice(cand)
                # simple feasibility check for the day (doesn't perfectly protect cascade but good approx)
                # we assume capacity constraint is roughly fine or we let objective penalize it (not implemented here)
                # Proper SA would repair, but we'll just check today's capacity.
                w_sum = sum(problem.wastes.get(n, 0.0) for n in rt)
                if w_sum + problem.wastes.get(v, 0.0) <= problem.capacity:
                    rt.append(v)
        else:
            # Drop a non-mandatory node
            cand = [v for v in rt if v not in problem.mandatory]
            if cand:
                v = self.rng.choice(cand)
                rt.remove(v)

        # opt it
        new_plan[day] = [two_opt(rt, problem.distance_matrix)]
        return new_plan

    def _run_multi_period_solver(
        self,
        problem: ProblemContext,
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """
        Execute the Simulated Annealing (SA) metaheuristic solver logic.

        SA is a probabilistic technique for approximating the global optimum of
        a given function. Specifically, it is a metaheuristic inspired by the
        metallurgical process of heating and controlled cooling to achieve a
        low-energy state.

        Search Logic:
        1. **Initial Solution**: Generated via greedy construction and 2-opt.
        2. **Neighbor Generation**: Proposes small, stochastic modifications to
           the multi-day plan (e.g., node swap/add/drop).
        3. **Acceptance Criterion**: Employs the Boltzmann-Metropolis condition.
           It always accepts improving moves and accepts deteriorating moves
           with a probability P = exp(delta / T), where delta is the loss in
           profit and T is the current temperature.
        4. **Cooling Schedule**: Systematically reduces T over time,
           transitioning the search from exploration to intensification.

        Args:
            problem: The current ProblemContext containing state data.
            multi_day_ctx: Optional context for spanning multiple rolling days.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
                - today_solution: Standardized solution context for Day 0.
                - full_plan: Collection plan (nested list by day and vehicle).
                - stats: Execution statistics (iterations, cost, profit).
        """
        np_rng = np.random.default_rng(self.seed)

        cur_plan = []
        cur_prob = problem
        for _d in range(problem.horizon):
            rt = greedy_day_route(cur_prob, np_rng)
            cur_plan.append([two_opt(rt, problem.distance_matrix)])
            cur_prob = cur_prob.advance(rt)

        best_plan = copy.deepcopy(cur_plan)
        best_prof = self._evaluate(best_plan, problem)
        cur_prof = best_prof

        temp = self.init_temp

        for _ in range(self.max_iter):
            nxt_plan = self._neighbor(cur_plan, problem)
            nxt_prof = self._evaluate(nxt_plan, problem)

            diff = nxt_prof - cur_prof
            if diff > 0 or self.rng.random() < math.exp(diff / max(1e-9, temp)):
                cur_plan = nxt_plan
                cur_prof = nxt_prof

                if cur_prof > best_prof:
                    best_plan = copy.deepcopy(cur_plan)
                    best_prof = cur_prof

            temp *= self.cooling_rate

        today_route = best_plan[0][0] if best_plan[0] else []
        sol = SolutionContext.from_problem(problem, today_route)

        return sol, best_plan, {"sa_iters": self.max_iter, "expected_profit": best_prof}
