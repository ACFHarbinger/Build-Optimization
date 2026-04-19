import copy
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import gurobipy as gp
import numpy as np

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.matheuristics.kernel_search.solver import (
    _dfj_subtour_elimination_callback,
    _reconstruct_tour,
    _setup_ks_model,
)
from logic.src.utils.policy.routes import route_cost, route_profit
from logic.src.utils.policy.wrappers import greedy_day_route

from .params import LNSMIPParams


@GlobalRegistry.register(
    PolicyTag.MATHEURISTIC,
    PolicyTag.LARGE_NEIGHBORHOOD_SEARCH,
    PolicyTag.MULTI_PERIOD,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("lns_mip")
class LNSMIPPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Large Neighborhood Search with MIP Repair.
    Destroys a subset of visits and repairs using an exact solver.
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.params = LNSMIPParams.from_config(config or {})
        self.rng = random.Random(self.params.seed)

    def _evaluate_plan(self, plan: List[List[List[int]]], problem: ProblemContext) -> float:
        """Evaluate expected multi-day profit deterministically (using mean increments)."""
        tot = 0.0
        cur_prob = problem
        for day_idx in range(problem.horizon):
            route = plan[day_idx][0] if plan[day_idx] else []
            tot += route_profit(route, cur_prob)
            cur_prob = cur_prob.advance(route)
        return tot

    def _destroy(self, problem: ProblemContext, plan: List[List[List[int]]]) -> Dict[int, List[int]]:  # noqa: C901
        d_destroy = min(self.params.d_destroy, problem.horizon)
        days = self.rng.sample(range(problem.horizon), d_destroy)

        strategy = self.rng.choice(["random", "proximity", "fill_level"])
        destroyed: Dict[int, List[int]] = {d: [] for d in days}
        k_per_day = max(1, self.params.k_destroy // d_destroy)

        if strategy == "random":
            for d in days:
                route = plan[d][0] if plan[d] else []
                if route:
                    rem = self.rng.sample(route, min(len(route), k_per_day))
                    destroyed[d] = rem
        elif strategy == "proximity":
            # Pick a random bin from a random route, remove nearest ones
            all_bins = set()
            for d in days:
                if plan[d]:
                    all_bins.update(plan[d][0])
            if all_bins:
                seed_bin = self.rng.choice(list(all_bins))
                d_row = problem.distance_matrix[seed_bin]
                # sort nodes by distance to seed
                closest = np.argsort(d_row)
                for d in days:
                    route = plan[d][0] if plan[d] else []
                    if route:
                        # find nearest in this route
                        r_set = set(route)
                        rem = []
                        for node in closest:
                            if len(rem) >= k_per_day:
                                break
                            if node in r_set:
                                rem.append(node)
                        destroyed[d] = rem
        elif strategy == "fill_level":
            # Just remove randomly for now to simplify, as real fill level depends on cascade
            for d in days:
                route = plan[d][0] if plan[d] else []
                if route:
                    rem = self.rng.sample(route, min(len(route), k_per_day))
                    destroyed[d] = rem

        return destroyed

    def _repair_mip(
        self, problem: ProblemContext, plan: List[List[List[int]]], destroyed: Dict[int, List[int]]
    ) -> List[List[List[int]]]:
        new_plan = copy.deepcopy(plan)
        cur_prob = problem

        # We must step through the horizon day by day since repair on day D affects day D+1.
        for d in range(problem.horizon):
            route = new_plan[d][0] if new_plan[d] else []
            if d in destroyed and destroyed[d]:
                rem_set = set(destroyed[d])
                fixed_bins = [v for v in route if v not in rem_set]
                fixed_load = sum(cur_prob.wastes.get(v, 0.0) for v in fixed_bins)
                avail_cap = max(0.0, cur_prob.capacity - fixed_load)

                # Setup KS model over removed_bins
                model = gp.Model("lns_repair")
                model.setParam("OutputFlag", 0)
                model.setParam("TimeLimit", self.params.mip_time_limit)
                model.setParam("MIPGap", self.params.mip_gap)
                model.Params.LazyConstraints = 1

                # intersection of problem mandatory and removed_bins (if any)
                mand_d = [v for v in cur_prob.mandatory if v in rem_set]

                # Filter wastes to only include removed bins
                sub_wastes = {v: cur_prob.wastes.get(v, 0.0) for v in rem_set}

                x, y = _setup_ks_model(
                    model,
                    cur_prob.distance_matrix,
                    sub_wastes,
                    avail_cap,
                    cur_prob.revenue_per_kg,
                    cur_prob.cost_per_km,
                    mand_d,
                    True,
                )

                model.optimize(_dfj_subtour_elimination_callback)

                if model.SolCount > 0:
                    frag_tour, _ = _reconstruct_tour(len(cur_prob.distance_matrix), x, cur_prob.distance_matrix)
                    # Exclude depots
                    frag_tour = [v for v in frag_tour if v != 0]
                    # Simple merge: append repaired fragment
                    route = fixed_bins + frag_tour

                new_plan[d] = [route]

            cur_prob = cur_prob.advance(new_plan[d][0] if new_plan[d] else [])

        return new_plan

    def _accept(self, best_profit: float, new_profit: float, temp: float) -> bool:
        if new_profit > best_profit:
            return True
        if self.params.acceptance == "sa":
            diff = new_profit - best_profit
            return self.rng.random() < math.exp(diff / temp)
        return False

    def _run_multi_period_solver(
        self, problem: ProblemContext, multi_day_ctx: Optional[MultiDayContext]
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        # 1. Greedy initial
        full_plan = []
        cur_prob = problem
        np_rng = np.random.default_rng(self.params.seed)
        for _ in range(problem.horizon):
            rt = greedy_day_route(cur_prob, np_rng)
            full_plan.append([rt])
            cur_prob = cur_prob.advance(rt)

        best_plan = copy.deepcopy(full_plan)
        current_plan = full_plan
        best_profit = self._evaluate_plan(best_plan, problem)

        temp = self.params.sa_temperature

        for _iteration in range(self.params.max_iterations):
            destroyed = self._destroy(problem, current_plan)
            repaired = self._repair_mip(problem, current_plan, destroyed)
            new_prof = self._evaluate_plan(repaired, problem)

            if self._accept(best_profit, new_prof, temp):
                current_plan = repaired
                if new_prof > best_profit:
                    best_plan = copy.deepcopy(repaired)
                    best_profit = new_prof

            if self.params.acceptance == "sa":
                temp *= self.params.sa_cooling

        today_route = best_plan[0][0] if best_plan[0] else []
        sol = SolutionContext.from_single_route(
            route=today_route,
            profit=route_profit(today_route, problem),
            cost=route_cost(today_route, problem),
            metadata={"iterations": self.params.max_iterations},
        )
        return sol, best_plan, {"lns_iterations": self.params.max_iterations}
