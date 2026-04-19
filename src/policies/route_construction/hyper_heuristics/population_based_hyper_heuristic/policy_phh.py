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
from logic.src.policies.route_construction.hyper_heuristics.population_based_hyper_heuristic.params import PHHParams
from logic.src.utils.policy.llh_pool import LLHPool
from logic.src.utils.policy.routes import (
    route_profit,
)
from logic.src.utils.policy.wrappers import (
    greedy_day_route,
)


@GlobalRegistry.register(
    PolicyTag.HYPER_HEURISTIC,
    PolicyTag.POPULATION_BASED,
    PolicyTag.MULTI_PERIOD,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("phh")
class PHHPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Population-based Hyper-Heuristic (PHH).

    PHH generalizes the concept of a selection hyper-heuristic by maintaining
    a population of solution/LLH-manager pairs. Each individual in the
    population independently explores the search space using its own internal
    selection logic, but periodic interactions (cross-pollination or
    tournament replacement) allow high-performing operator sequences or
    solutions to propagate through the population.

    Search Logic:
    1. **Initialization**: Generates a diverse population of multi-period
       collection plans using randomized greedy construction.
    2. **Independent Search**: Each individual undergoes multiple rounds of
       stochastic improvement using Low-Level Heuristics (LLHs).
    3. **Interaction**: Periodically performs tournament selection or
       exchange to replace lower-performing individuals with copies or
       recombinations of high-performing ones.
    4. **Diversity Management**: Prevents premature convergence by ensuring the
       LLH application remains stochastic.

    Registry key: ``"phh"``
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.params = PHHParams.from_config(config)
        self.pop_size = self.params.pop_size
        self.gens = self.params.gens
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
        Execute the Population-based Hyper-Heuristic solver.

        Args:
            problem: The current ProblemContext containing state data.
            multi_day_ctx: Optional context for spanning multiple rolling days.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
                - today_solution: Standardized solution context for Day 0.
                - best_plan: The highest-performing multi-day collection plan.
                - stats: Execution statistics (generations, best fitness).
        """
        np_rng = np.random.default_rng(self.seed)

        pop = []
        # Initialize with a valid empty structure to satisfy Mypy
        best_plan: List[List[List[int]]] = [[] for _ in range(problem.horizon)]
        best_prof = -float("inf")

        # Init pop
        for _ in range(self.pop_size):
            cur_plan = []
            cur_prob = problem
            for _d in range(problem.horizon):
                rt = greedy_day_route(cur_prob, np_rng)
                cur_plan.append([rt])
                cur_prob = cur_prob.advance(rt)
            pop.append(cur_plan)

        for _ in range(self.gens):
            for i in range(self.pop_size):
                p = pop[i]

                # Apply random LLH
                llh = self.rng.choice(self.llhs)
                p_new = llh(p, problem, self.rng)

                prof_old = self._evaluate(p, problem)
                prof_new = self._evaluate(p_new, problem)

                if prof_new > prof_old:
                    pop[i] = p_new
                    if prof_new > best_prof:
                        best_prof = prof_new
                        best_plan = copy.deepcopy(p_new)

            # Simplistic cross-pollination (tournament replacement)
            fits = [self._evaluate(x, problem) for x in pop]
            worst_idx = np.argmin(fits)
            best_idx = np.argmax(fits)
            pop[worst_idx] = copy.deepcopy(pop[best_idx])

        today_route = best_plan[0][0] if best_plan and best_plan[0] else []
        sol = SolutionContext.from_problem(problem, today_route)

        return sol, best_plan, {"phh_gens": self.gens, "expected_profit": best_prof}
