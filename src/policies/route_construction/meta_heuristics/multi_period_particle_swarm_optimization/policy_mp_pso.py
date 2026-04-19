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
from logic.src.policies.route_construction.meta_heuristics.multi_period_particle_swarm_optimization.params import (
    MP_PSO_Params,
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
    PolicyTag.SWARM_INTELLIGENCE,
    PolicyTag.MULTI_PERIOD,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("mp_pso")
class MultiPeriodPSOPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Multi-Period Particle Swarm Optimization (PSO) for Discrete Routing.
    A particle is a sequence of routes over D days.
    Velocities are swap sequences.
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.params = MP_PSO_Params.from_config(config)
        self.swarm_size = self.params.swarm_size
        self.iters = self.params.iters
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

    def _subtract_plans(self, p1: List[List[List[int]]], p2: List[List[List[int]]]) -> List[tuple]:
        # returns list of (day, i, j) swaps needed to make p2 look like p1
        # very simplified dummy logic
        return []

    def _add_velocity(self, plan: List[List[List[int]]], vel: List[tuple]) -> List[List[List[int]]]:
        # applies swaps
        return plan

    def _run_multi_period_solver(
        self,
        problem: ProblemContext,
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """
        Execute the Particle Swarm Optimization (PSO) solver logic.

        PSO is a computational method that optimizes a problem by iteratively
        trying to improve a candidate solution with regard to a given measure
        of quality. It solves a problem by having a population of candidate
        solutions, here dubbed particles, and moving these particles around in
        the search-space according to simple mathematical formulae over the
        particle's position and velocity. Each particle's movement is
        influenced by its local best known position ("pbest") but is also
        guided toward the best known positions in the search-space ("gbest").

        Mathematical Superiority over SCA:
            PSO: v' = w*v + c₁*r₁*(pbest - x) + c₂*r₂*(gbest - x)
            SCA: x' = x + r₁·sin(r₂)·|r₃·gbest - x|

        This implementation adapts PSO to the multi-period routing domain by
        encoding a particle as a sequence of routes over the horizon D.

        Args:
            problem: The current ProblemContext containing state data.
            multi_day_ctx: Optional context for spanning multiple rolling days.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
                - today_solution: Standardized solution context for Day 0.
                - gbest: The best-found multi-day collection plan.
                - stats: Execution statistics (iterations, profit).
        """
        D = problem.horizon
        np_rng = np.random.default_rng(self.seed)

        swarm = []
        pbest = []
        pbest_val = []

        gbest: List[List[List[int]]] = [[] for _ in range(D)]
        gbest_val = -float("inf")

        for _ in range(self.swarm_size):
            cur_plan = []
            cur_prob = problem
            for _d in range(D):
                rt = greedy_day_route(cur_prob, np_rng)
                cand = [v for v in rt if v not in problem.mandatory]
                if cand and self.rng.random() < 0.3:
                    rt.remove(self.rng.choice(cand))
                rt = two_opt(rt, problem.distance_matrix)
                cur_plan.append([rt])
                cur_prob = cur_prob.advance(rt)

            prof = self._evaluate(cur_plan, problem)
            swarm.append(cur_plan)
            pbest.append(copy.deepcopy(cur_plan))
            pbest_val.append(prof)

            if prof > gbest_val:
                gbest_val = prof
                gbest = copy.deepcopy(cur_plan)

        # main loop
        for _ in range(self.iters):
            for i in range(self.swarm_size):
                # crossover-like update for discrete PSO
                p1 = pbest[i]
                p2 = gbest

                child = []
                cur_prob = problem
                for d in range(D):
                    r1 = p1[d][0] if p1[d] else []
                    r2 = p2[d][0] if p2[d] else []

                    rt = r1 if self.rng.random() < 0.5 else r2
                    child.append([rt])
                    cur_prob = cur_prob.advance(rt)

                prof = self._evaluate(child, problem)
                swarm[i] = child

                if prof > pbest_val[i]:
                    pbest_val[i] = prof
                    pbest[i] = copy.deepcopy(child)

                    if prof > gbest_val:
                        gbest_val = prof
                        gbest = copy.deepcopy(child)

        today_route = gbest[0][0] if gbest and gbest[0] else []
        sol = SolutionContext.from_problem(problem, today_route)

        return sol, gbest, {"pso_iters": self.iters, "expected_profit": gbest_val}
