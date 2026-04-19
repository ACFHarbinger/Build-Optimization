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
from logic.src.policies.route_construction.meta_heuristics.multi_period_iterated_local_search.params import (
    MP_ILS_Params,
)
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
@RouteConstructorRegistry.register("mp_ils")
class MultiPeriodILSPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Multi-Period Iterated Local Search (ILS) metaheuristic.
    1. Initial solution (greedy + 2-opt)
    2. Local search (steepest descent)
    3. Perturbation (destroy and recreate)
    4. Acceptance criterion
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.params = MP_ILS_Params.from_config(config)
        self.max_iter = self.params.iters
        self.perturb_size = self.params.perturb_size
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

    def _local_search(self, plan: List[List[List[int]]], problem: ProblemContext) -> List[List[List[int]]]:
        # Apply 2-opt to each day
        new_plan = []
        cur_prob = problem
        for d in range(problem.horizon):
            rt = plan[d][0] if plan[d] else []
            opt_rt = two_opt(list(rt), cur_prob.distance_matrix)
            new_plan.append([opt_rt])
            cur_prob = cur_prob.advance(opt_rt)
        return new_plan

    def _perturb(self, plan: List[List[List[int]]], problem: ProblemContext) -> List[List[List[int]]]:
        # Destroys `perturb_size` nodes and tries to re-insert them
        # (or just simple random drop for a VRPP where we can drop)
        new_plan = copy.deepcopy(plan)
        cur_prob = problem

        for d in range(problem.horizon):
            rt = new_plan[d][0] if new_plan[d] else []
            if not rt:
                cur_prob = cur_prob.advance(rt)
                continue

            n_remove = min(self.perturb_size, len(rt))
            rem = set(self.rng.sample(rt, n_remove))
            rt = [v for v in rt if v not in rem]

            # recreate phase: optionally try greedy insertion of non-mandatory
            # (In VRPP, we don't have to re-insert if it's not mandatory)
            mand_rem = rem.intersection(set(cur_prob.mandatory))
            for m in mand_rem:
                rt.append(m)  # force append mandatory at end, we'll LS it later

            new_plan[d] = [rt]
            cur_prob = cur_prob.advance(rt)

        return new_plan

    def _run_multi_period_solver(
        self,
        problem: ProblemContext,
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """
        Execute the Iterated Local Search (ILS) metaheuristic solver logic.

        ILS is a simple yet powerful metaheuristic that iteratively applies a
        local search to a perturbed solution. It operates in a loop:
        1. **Initial Solution**: A greedy construction followed by local refinement.
        2. **Perturbation**: The current solution is modified (e.g., via ruin-and-recreate
           or random moves) to escape the current local optimum.
        3. **Local Search**: The perturbed solution is refined using local search
           operators (such as 2-opt) until a new local optimum is found.
        4. **Acceptance**: The new local optimum is accepted if it improves upon
           the current or best-known objective value.

        In this multi-period implementation, the search optimizes a schedule
        of routes over the provides horizon T, accounting for inventory
        transitions and expected profits across multiple days.

        Args:
            problem: The current ProblemContext containing state data.
            multi_day_ctx: Optional context for spanning multiple rolling days.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
                - today_solution: Standardized solution context for Day 0.
                - full_plan: Collection plan for all days in the horizon.
                - stats: Execution statistics (iterations, cost, profit).
        """
        D = problem.horizon
        np_rng = np.random.default_rng(self.seed)

        # Initial
        cur_plan = []
        cur_prob = problem
        for _d in range(D):
            rt = greedy_day_route(cur_prob, np_rng)
            cur_plan.append([rt])
            cur_prob = cur_prob.advance(rt)

        cur_plan = self._local_search(cur_plan, problem)
        best_plan = copy.deepcopy(cur_plan)
        best_prof = self._evaluate(best_plan, problem)
        cur_prof = best_prof

        for _ in range(self.max_iter):
            perturbed = self._perturb(cur_plan, problem)
            improved = self._local_search(perturbed, problem)

            imp_prof = self._evaluate(improved, problem)

            # Acceptance (simple greedy)
            if imp_prof > cur_prof:
                cur_plan = copy.deepcopy(improved)
                cur_prof = imp_prof

                if imp_prof > best_prof:
                    best_plan = copy.deepcopy(improved)
                    best_prof = imp_prof

        today_route = best_plan[0][0] if best_plan[0] else []
        sol = SolutionContext.from_problem(problem, today_route)

        return sol, best_plan, {"ils_iters": self.max_iter, "expected_profit": best_prof}
