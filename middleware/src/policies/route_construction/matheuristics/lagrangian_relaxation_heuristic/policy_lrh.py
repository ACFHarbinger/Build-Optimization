import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.matheuristics.lagrangian_relaxation_heuristic.params import LRHParams
from logic.src.utils.policy.routes import (
    route_cost,
    route_profit,
)
from logic.src.utils.policy.wrappers import (
    greedy_day_route,
)


@GlobalRegistry.register(
    PolicyTag.MATHEURISTIC,
    PolicyTag.DECOMPOSITION,
    PolicyTag.MULTI_PERIOD,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("lrh")
class LagrangianRelaxationHeuristicPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Lagrangian Relaxation Heuristic.
    Relaxes capacity constraints into the objective via Lagrangian multipliers.
    Resolves simplified subproblems and uses a subgradient method to tune multipliers.
    Converts relaxed solutions to feasible using greedy repair.
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.params = LRHParams.from_config(config)
        self.max_iter = self.params.max_iter
        self.step_size_initial = self.params.step_size
        self.halving_freq = self.params.halving_freq
        self.seed = self.params.seed

    def _solve_relaxed_subproblem(self, problem: ProblemContext, lambdas: List[float]) -> List[List[List[int]]]:
        # This is a very simplistic relaxation purely for structural demonstration.
        # It treats the routing without capacity constraints, effectively solving
        # a prize-collecting TSP (or multiple) per day, where prize is modified by lambda * waste.
        # For feasibility, we will just use our greedy construct but shift the revenues by lambda.

        # Real Lagrangian would solve exact DP or MIP for the relaxed problem.
        # Since unconstrained PCTSP is still NP-hard, we approximate via our greedy operator
        # but with modified revenue ratios.

        sim_plan: List[List[List[int]]] = []
        cur_prob = problem
        rng = np.random.default_rng(self.seed)
        for d in range(problem.horizon):
            lam = lambdas[d]

            # modify revenue for day d using problem.replace
            # P(A) = revenue * sum(w) - c * dist - lambda * (sum(w) - Q)
            # effectively, new revenue per kg is (revenue - lambda)
            # greedy construct relies on problem.revenue_per_kg
            new_r = cur_prob.revenue_per_kg - lam

            if new_r <= 0:
                # no profit to be made, empty route
                sim_plan.append([[]])
            else:
                # We can't replace fields in ProblemContext because it has frozen dynamics, but we can override inside kwargs or use the internal method.
                mod_prob = ProblemContext(
                    distance_matrix=cur_prob.distance_matrix,
                    wastes=cur_prob.wastes,
                    fill_rate_means=cur_prob.fill_rate_means,
                    fill_rate_stds=cur_prob.fill_rate_stds,
                    capacity=cur_prob.capacity,  # even if relaxed, greedy bound acts as upper bound
                    max_fill=cur_prob.max_fill,
                    revenue_per_kg=new_r,
                    cost_per_km=cur_prob.cost_per_km,
                    mandatory=cur_prob.mandatory,
                    locations=cur_prob.locations,
                    horizon=cur_prob.horizon - d,
                    day_index=cur_prob.day_index,
                )
                rt = greedy_day_route(mod_prob, rng)
                sim_plan.append([rt])

            cur_prob = cur_prob.advance(sim_plan[-1][0])

        return sim_plan

    def _repair_to_feasible(self, problem: ProblemContext, plan: List[List[List[int]]]) -> List[List[List[int]]]:
        """Repair capacity violations by dropping least profitable nodes."""
        feas_plan = []
        cur_prob = problem
        for d in range(problem.horizon):
            route = plan[d][0] if plan[d] else []
            r_set = set(route)

            total_w = sum(cur_prob.wastes.get(v, 0.0) for v in r_set)

            # Simple repair: while over capacity, remove the node with lowest w / dist
            if total_w > cur_prob.capacity:
                # greedy drop
                rt_copy = list(route)
                while sum(cur_prob.wastes.get(v, 0.0) for v in rt_copy) > cur_prob.capacity and rt_copy:
                    # find worst node to drop (not mandatory)
                    worst_idx = -1
                    worst_val = float("inf")
                    for r_i, v in enumerate(rt_copy):
                        if v in cur_prob.mandatory:
                            continue
                        val = cur_prob.wastes.get(v, 0.0)
                        if val < worst_val:
                            worst_idx = r_i
                            worst_val = val
                    if worst_idx == -1:
                        break  # Only mandatory left
                    rt_copy.pop(worst_idx)
                route = rt_copy

            feas_plan.append([route])
            cur_prob = cur_prob.advance(route)

        return feas_plan

    def _evaluate(self, plan, problem):
        tot = 0.0
        cur_prob = problem
        for idx in range(problem.horizon):
            rt = plan[idx][0] if plan[idx] else []
            tot += route_profit(rt, cur_prob)
            cur_prob = cur_prob.advance(rt)
        return tot

    def _run_multi_period_solver(
        self, problem: ProblemContext, multi_day_ctx: Optional[MultiDayContext]
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        D = problem.horizon
        lambdas = [0.0] * D
        pi = self.step_size_initial

        best_feas_plan = None
        best_feas_val = -float("inf")

        for k in range(self.max_iter):
            # Solve relaxed
            relaxed_plan = self._solve_relaxed_subproblem(problem, lambdas)

            # Subgradient approx
            cur_prob = problem
            grads = [0.0] * D
            for d in range(D):
                route = relaxed_plan[d][0] if relaxed_plan[d] else []
                total_w = sum(cur_prob.wastes.get(v, 0.0) for v in route)
                grads[d] = total_w - cur_prob.capacity
                cur_prob = cur_prob.advance(route)

            # Repair
            feas_plan = self._repair_to_feasible(problem, relaxed_plan)
            feas_val = self._evaluate(feas_plan, problem)

            if feas_val > best_feas_val:
                best_feas_val = feas_val
                best_feas_plan = copy.deepcopy(feas_plan)

            # Update lambdas
            sum_sq_grad = sum(g**2 for g in grads)
            if sum_sq_grad > 1e-6:
                # we don't have optimal LB easily, just step back
                step = pi / math.sqrt(sum_sq_grad)
                for d in range(D):
                    lambdas[d] = max(0.0, lambdas[d] + step * grads[d])

            if (k + 1) % self.halving_freq == 0:
                pi /= 2.0

        if not best_feas_plan:
            # fallback
            cur_prob = problem
            best_feas_plan = []
            rng = np.random.default_rng(self.seed)
            for _ in range(problem.horizon):
                rt = greedy_day_route(cur_prob, rng)
                best_feas_plan.append([rt])
                cur_prob = cur_prob.advance(rt)

        today_route = best_feas_plan[0][0] if best_feas_plan[0] else []
        sol = SolutionContext.from_single_route(
            route=today_route,
            profit=route_profit(today_route, problem),
            cost=route_cost(today_route, problem),
            metadata={},
        )
        return sol, best_feas_plan, {"lh_iters": self.max_iter}
