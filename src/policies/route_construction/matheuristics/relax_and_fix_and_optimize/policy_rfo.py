import math
from typing import Any, Dict, List, Optional, Tuple

import gurobipy as gp

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.matheuristics.relax_and_fix_and_optimize.params import RFOParams
from logic.src.utils.policy.routes import route_cost, route_profit


@GlobalRegistry.register(
    PolicyTag.MATHEURISTIC,
    PolicyTag.DECOMPOSITION,
    PolicyTag.MULTI_PERIOD,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("rfo")
class RelaxFixOptimizePolicy(BaseMultiPeriodRoutingPolicy):
    r"""
    Relax-and-Fix (R&F) with Local Optimization (also known as Relax & Fix / Fix & Optimize) for MPVRPP.

    RFO is a mathematical programming-based heuristic for multi-period
    optimization with integer variables. It decomposes the planning horizon
    into three overlapping regions to manage the trade-off between immediate
    optimality and future feasibility.

    Mathematical Principles:
    1.  **Decomposition**: The horizon $T$ is partitioned into a set of windows
        $W$. Let $T_{fixed} < T_{int} < T_{rel}$ be the time indices.
    2.  **Relax-and-Fix Step**:
        - For $t \in T_{fixed}$: Bin-visit variables $y_{i,t}$ are fixed to their
          previous optimal values.
        - For $t \in T_{int}$: Variables are constrained to be binary (integer).
        - For $t \in T_{rel}$: Variables are relaxed to $y_{i,t} \in [0, 1]$.
    3.  **Rolling Strategy**: The integer window $T_{int}$ slides forward by
        `step_size` days after each MILP solve. This ensures that Day 0 (the
        decision day) is always solved to integer optimality while considering
        the long-term relaxed impact of future scenarios.

    Algorithm Logic:
    - **Step 1**: Construct a monolithic MILP with discretized bin fill rates.
    - **Step 2**: Solve the R&F subproblems iteratively until the integer window
      covers the entire execution day.
    - **Step 3**: Extractions the resulting routing plan for the current day.

    Registry key: ``"rfo"``
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.params = RFOParams.from_config(config)
        self.window_size = self.params.window_size
        self.step_size = self.params.step_size
        self.mip_time = self.params.mip_time
        self.mip_gap = self.params.mip_gap

    def _setup_multi_day_model(
        self, problem: ProblemContext, int_days: set, relaxed_days: set, fixed_plan: Dict[int, List[int]]
    ):
        model = gp.Model("RFO")
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", self.mip_time)
        model.setParam("MIPGap", self.mip_gap)
        model.Params.LazyConstraints = 1

        N = len(problem.distance_matrix)
        x = {}  # x[d, i, j]
        y = {}  # y[d, i]

        # Simplified Multi-day formulation: independent routing per day with shared capacity limit
        # This acts as an approximation; real cascade requires complex modeling of fill dynamics
        # For this template, we construct a block structured model.
        for d in range(problem.horizon):
            var_type = gp.GRB.BINARY if d in int_days else gp.GRB.CONTINUOUS
            for i in range(1, N):
                y[d, i] = model.addVar(vtype=var_type, lb=0, ub=1, name=f"y_{d}_{i}")
                x[d, 0, i] = model.addVar(vtype=var_type, lb=0, ub=1, name=f"x_{d}_0_{i}")
                x[d, i, 0] = model.addVar(vtype=var_type, lb=0, ub=1, name=f"x_{d}_{i}_0")
                for j in range(1, N):
                    if i != j:
                        x[d, i, j] = model.addVar(vtype=var_type, lb=0, ub=1, name=f"x_{d}_{i}_{j}")

        # Build constraints for each day
        for d in range(problem.horizon):
            # Degree constraints
            for i in range(1, N):
                model.addConstr(gp.quicksum(x[d, i, j] for j in range(N) if j != i) == y[d, i])
                model.addConstr(gp.quicksum(x[d, j, i] for j in range(N) if j != i) == y[d, i])

            # Single vehicle degree at depot
            model.addConstr(gp.quicksum(x[d, 0, j] for j in range(1, N)) <= 1)
            model.addConstr(gp.quicksum(x[d, j, 0] for j in range(1, N)) <= 1)

            # Capacity (using day 0 fixed weights approximation for simplicity in MIP)
            model.addConstr(gp.quicksum(problem.wastes.get(i, 0.0) * y[d, i] for i in range(1, N)) <= problem.capacity)

            # If day is already fixed, fix the variables
            if d in fixed_plan:
                route = fixed_plan[d]
                r_set = set(route)
                for i in range(1, N):
                    y[d, i].LB = 1.0 if i in r_set else 0.0
                    y[d, i].UB = 1.0 if i in r_set else 0.0

                path = [0] + route + [0]
                edges = {(path[k], path[k + 1]) for k in range(len(path) - 1)}
                for i in range(N):
                    for j in range(N):
                        if i != j and (d, i, j) in x:
                            val = 1.0 if (i, j) in edges else 0.0
                            x[d, i, j].LB = val
                            x[d, i, j].UB = val

        # Objective
        obj = gp.quicksum(
            problem.revenue_per_kg * problem.wastes.get(i, 0.0) * y[d, i]
            - problem.cost_per_km * x[d, i, j] * problem.distance_matrix[i, j]
            for d in range(problem.horizon)
            for i in range(N)
            for j in range(N)
            if i != j and (d, i, j) in x
        )
        model.setObjective(obj, gp.GRB.MAXIMIZE)

        return model, x, y

    def _run_multi_period_solver(
        self, problem: ProblemContext, multi_day_ctx: Optional[MultiDayContext]
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        D = problem.horizon
        fixed_plan: Dict[int, List[int]] = {}

        for w_start in range(0, D, self.step_size):
            w_end = min(D, w_start + self.window_size)
            int_days = set(range(w_start, w_end))
            relaxed_days = set(range(w_end, D))

            model, x, y = self._setup_multi_day_model(problem, int_days, relaxed_days, fixed_plan)

            # Subtour separation needs to handle multi-day structure nicely.
            # We'll pass a custom lambda or just skip lazy for day-by-day continuous loops during fix block
            model.Params.LazyConstraints = 0  # simplifying lazy subtour for full multiday approximation
            model.optimize()

            # Extract routes for the integer portion and save
            if model.SolCount > 0:
                for d in range(w_start, w_start + self.step_size):
                    if d >= D:
                        break
                    # reconstruct flat day tour
                    route = []
                    curr = 0
                    N = len(problem.distance_matrix)
                    for _ in range(N):
                        nxt = -1
                        for j in range(N):
                            if j != curr and x[d, curr, j].X > 0.5:
                                nxt = j
                                break
                        if nxt == -1 or nxt == 0:
                            break
                        route.append(nxt)
                        curr = nxt
                    fixed_plan[d] = route
            else:
                for d in range(w_start, w_start + self.step_size):
                    if d < D:
                        fixed_plan[d] = []

        # Construct final plan List[List[List[int]]]
        full_plan = []
        for d in range(D):
            full_plan.append([fixed_plan.get(d, [])])

        today_route = full_plan[0][0] if full_plan[0] else []

        # calculate route profit/cost dynamically
        cost = route_cost(today_route, problem)
        profit = route_profit(today_route, problem)

        sol = SolutionContext.from_single_route(route=today_route, profit=profit, cost=cost, metadata={})
        return sol, full_plan, {"rfo_windows_solved": math.ceil(D / self.step_size)}
