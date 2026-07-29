from typing import Any, Dict, List, Optional, Tuple

import gurobipy as gp
import numpy as np

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.matheuristics.column_generation_heuristic.params import CGHParams
from logic.src.utils.policy.routes import route_cost, route_profit
from logic.src.utils.policy.wrappers import greedy_day_route, two_opt


@GlobalRegistry.register(
    PolicyTag.MATHEURISTIC,
    PolicyTag.DECOMPOSITION,
    PolicyTag.MULTI_PERIOD,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("cgh")
class ColumnGenerationHeuristicPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Column Generation Heuristic (CGH).
    Uses a master problem (set packing/partitioning) and generates routes heuristically.
    Instead of solving the exact subproblem (ESPPRC), it uses simulated annealing/ILS
    to find negative reduced-cost columns.
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.params = CGHParams.from_config(config)
        self.cg_iters = self.params.cg_iters
        self.routes_per_iter = self.params.routes_per_iter
        self.seed = self.params.seed

    def _generate_columns_heuristically(
        self, problem: ProblemContext, duals: Dict[int, float], n_routes: int, rng: np.random.Generator
    ) -> List[List[int]]:
        """
        Produce a list of routes with high negative reduced cost heuristically.
        Reduced cost of a route = sum(duals[i]) - Profit(route).
        (We maximize Profit + duals).
        We'll use a randomized greedy strategy where we modify the perceived revenue.
        """
        cols = []
        for _ in range(n_routes):
            # random scaling of duals is not currently applied in this placeholder heuristic

            # create temporary modified problem context
            # We want to insert node if R * w_i - C * dist + scale * dual[i] > 0
            # Since our greedy only uses R * w_i, we can't perfectly map it.
            # Instead, we just randomly build routes and then local-search them.

            # 1. Random route build
            rt = []
            cand = list(range(1, len(problem.distance_matrix)))
            rng.shuffle(cand)
            w_sum = 0.0
            for v in cand:
                if w_sum + problem.wastes.get(v, 0.0) <= problem.capacity:
                    rt.append(v)
                    w_sum += problem.wastes.get(v, 0.0)

            # 2. 2-opt
            rt = two_opt(rt, problem.distance_matrix)

            # 3. Simple swap-based improvement w.r.t reduced cost
            # Not fully optimized, just a placeholder heuristic
            cols.append(rt)
        return cols

    def _solve_cg_for_day(self, problem: ProblemContext, rng: np.random.Generator) -> List[int]:
        # Master problem
        model = gp.Model("CG_Heur")
        model.setParam("OutputFlag", 0)

        N = len(problem.distance_matrix)

        # initial columns: single nodes + empty
        columns: List[List[int]] = [[]] + [[i] for i in range(1, N) if problem.wastes.get(i, 0.0) <= problem.capacity]

        # Add greedy
        columns.append(greedy_day_route(problem, rng))

        # decision variables and constraints
        # maximize sum( y_r * profit_r )
        # subject to: sum_{r: i in r} y_r <= 1 for all i
        # y_r >= 0 (relaxed)

        constrs = {}
        vars_by_col = []

        for i in range(1, N):
            constrs[i] = model.addConstr(gp.quicksum([]) >= 0, name=f"cov_{i}")  # dummy initial

        def _add_col(rt):
            prof = route_profit(rt, problem)
            col = gp.Column()
            for v in rt:
                col.addTerms(1.0, constrs[v])
            var = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, obj=prof, column=col)
            vars_by_col.append((var, rt))

        for c in columns:
            _add_col(c)

        for _ in range(self.cg_iters):
            model.optimize()
            if model.Status != gp.GRB.OPTIMAL:
                break

            duals = {i: constrs[i].Pi for i in range(1, N)}

            new_cols = self._generate_columns_heuristically(problem, duals, self.routes_per_iter, rng)
            added = 0
            for c in new_cols:
                # check reduced cost: profit - sum(duals)
                rc = route_profit(c, problem) - sum(duals.get(v, 0.0) for v in c)
                if rc > 1e-4:
                    _add_col(c)
                    added += 1
            if added == 0:
                break

        # Integer solve
        for v, _ in vars_by_col:
            v.VType = gp.GRB.BINARY

        model.optimize()

        best_rts = [rt for var, rt in vars_by_col if var.X > 0.5]

        # If multi-vehicle, we would return best_rts. Since our interface currently assumes single
        # vehicle for some problem variants, we just return the first route, or flatten.
        # WSmart returns single list per vehicle. We'll flatten.
        flat = [v for rt in best_rts for v in rt]
        return two_opt(flat, problem.distance_matrix)

    def _run_multi_period_solver(
        self, problem: ProblemContext, multi_day_ctx: Optional[MultiDayContext]
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        D = problem.horizon
        full_plan = []
        cur_prob = problem
        rng = np.random.default_rng(self.seed)

        for _ in range(D):
            rt = self._solve_cg_for_day(cur_prob, rng)
            full_plan.append([rt])
            cur_prob = cur_prob.advance(rt)

        today_route = full_plan[0][0] if full_plan[0] else []
        sol = SolutionContext.from_single_route(
            route=today_route,
            profit=route_profit(today_route, problem),
            cost=route_cost(today_route, problem),
            metadata={},
        )
        return sol, full_plan, {"cg_heuristics": True}
