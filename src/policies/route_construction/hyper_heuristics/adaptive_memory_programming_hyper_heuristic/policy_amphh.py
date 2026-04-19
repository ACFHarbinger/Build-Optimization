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
from logic.src.policies.route_construction.hyper_heuristics.adaptive_memory_programming_hyper_heuristic.params import (
    AMPHHParams,
)
from logic.src.utils.policy.llh_pool import LLHPool
from logic.src.utils.policy.routes import route_profit
from logic.src.utils.policy.wrappers import greedy_day_route


@GlobalRegistry.register(
    PolicyTag.HYPER_HEURISTIC,
    PolicyTag.POPULATION_BASED,
    PolicyTag.MULTI_PERIOD,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
    PolicyTag.ADAPTIVE_ALGORITHM,
)
@RouteConstructorRegistry.register("amphh")
class AMPHHPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Adaptive Memory Programming Hyper-Heuristic (AMPHH).

    AMPHH is a search methodology that bridges the gap between memory-based
    metaheuristics (like Tabu Search) and hyper-heuristics. It maintains an
    "adaptive memory" of high-quality solution components (e.g., specific
    daily routes) found during the search and periodically reconstructs new
    multi-period plans by sampling from this memory.

    Search Logic:
    1. **Memory Initialization**: Populates the adaptive memory with initial
       greedy solutions and their stochastic perturbations.
    2. **Reconstruction**: Builds a new multi-period plan by selecting high-quality
       routes from memory for each day in the horizon.
    3. **LLH Refinement**: Applies a randomly selected Low-Level Heuristic (LLH)
       to refine the reconstructed plan, promoting intensification.
    4. **Memory Update**: Updates the adaptive memory based on the quality of
       the newly found solution, maintaining diversity and elite coverage.

    Registry key: ``"amphh"``
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.params = AMPHHParams.from_config(config)
        self.mem_size = self.params.mem_size
        self.iters = self.params.iters
        self.seed = self.params.seed
        self.rng = random.Random(self.seed)
        self.llhs = LLHPool.get_all()
        # Memory structure: stores evaluated entire plans
        self.memory: List[Tuple[float, List[List[List[int]]]]] = []

    def _evaluate(self, plan: List[List[List[int]]], problem: ProblemContext) -> float:
        tot = 0.0
        cur_prob = problem
        for d in range(problem.horizon):
            rt = plan[d][0] if plan[d] else []
            tot += route_profit(rt, cur_prob)
            cur_prob = cur_prob.advance(rt)
        return tot

    def _update_memory(self, prof: float, plan: List[List[List[int]]]):
        self.memory.append((prof, copy.deepcopy(plan)))
        self.memory.sort(key=lambda x: x[0], reverse=True)
        if len(self.memory) > self.mem_size:
            self.memory.pop()

    def _run_multi_period_solver(
        self,
        problem: ProblemContext,
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """
        Execute the Adaptive Memory Programming Hyper-Heuristic solver.

        Args:
            problem: The current ProblemContext containing state data.
            multi_day_ctx: Optional context for spanning multiple rolling days.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
                - today_solution: Standardized solution context for Day 0.
                - best_plan: The highest-performing multi-day collection plan.
                - stats: Execution statistics (iterations, memory updates).
        """
        np_rng = np.random.default_rng(self.seed)

        # Initialize memory with some greedy runs + perturbations
        for _ in range(self.mem_size):
            cur_plan = []
            cur_prob = problem
            for _d in range(problem.horizon):
                rt = greedy_day_route(cur_prob, np_rng)
                cand = [v for v in rt if v not in problem.mandatory]
                if cand and self.rng.random() < 0.5:
                    rt.remove(self.rng.choice(cand))  # mild perturb
                cur_plan.append([rt])
                cur_prob = cur_prob.advance(rt)
            prof = self._evaluate(cur_plan, problem)
            self._update_memory(prof, cur_plan)

        best_plan = self.memory[0][1]
        best_prof = self.memory[0][0]

        for _ in range(self.iters):
            # Combine components from memory uniformly at random for each day
            child = []
            cur_prob = problem
            for d in range(problem.horizon):
                parent_idx = self.rng.randint(0, len(self.memory) - 1)
                parent_plan = self.memory[parent_idx][1]
                rt = parent_plan[d][0] if parent_plan[d] else []
                child.append([rt])
                cur_prob = cur_prob.advance(rt)

            # Apply LLH
            llh = self.rng.choice(self.llhs)
            child = llh(child, problem, self.rng)

            prof = self._evaluate(child, problem)
            self._update_memory(prof, child)

            if prof > best_prof:
                best_prof = prof
                best_plan = copy.deepcopy(child)

        today_route = best_plan[0][0] if best_plan[0] else []
        sol = SolutionContext.from_problem(problem, today_route)

        return sol, best_plan, {"amp_iters": self.iters, "expected_profit": best_prof}
