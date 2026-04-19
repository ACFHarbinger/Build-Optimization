r"""
Exact Stochastic Dynamic Programming (ESDP) Policy Adapter for Stochastic VRPP.

ESDP provides the exact optimal solution for small-scale SIRP instances by
exhaustively traversing the state space and solving the Recursive Bellman
Equation over the horizon $T$.

Mathematical Formulation:
    Let $S_t$ be the state at day $t$, representing the inventory levels of
    all bins. The value function $V_t(S_t)$ is defined as:

        V_t(S_t) = max_{a_t \in \mathcal{A}} { R(S_t, a_t) - C(a_t) +
                   E[ V_{t+1}(S_{t+1} | S_t, a_t) ] }

    where:
        - $a_t$: Routing action (subset of bins to collect).
        - $R(S_t, a_t)$: Immediate revenue from collection.
        - $C(a_t)$: Immediate travel cost.
        - $E[...]$: Expected future profit based on transition probabilities.

Optimal Policy:
    At each step $t$ in a rolling horizon, ESDP retrieves the action $a_t^*$
    that maximizes the current value function given the observed inventory $S_t$.

State Space Discretization:
    To keep the DP table tractable, bin inventories are discretized into $L$
    levels (config: `discrete_levels`). For $N$ bins, the state space is $L^N$.
    Due to the exponential growth, ESDP is strictly limited to small instances
    (e.g., $N \le 10$).

Implementation:
    - **Offline Precomputation**: The DP table is computed once per instance
      during the first `execute()` call.
    - **Caching**: The `ExactSDPEngine` is cached globally to avoid
      recomputation across simulation days.

Registry key: ``"esdp"``
"""

import itertools
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from logic.src.configs.policies.esdp import ExactSDPConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.esdp_engine import (
    ExactSDPEngine,
)
from logic.src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.params import (
    SDPParams,
)

_SDP_CACHE: Dict[Tuple[int, int, int, float], ExactSDPEngine] = {}


@GlobalRegistry.register(
    PolicyTag.EXACT,
    PolicyTag.STOCHASTIC,
    PolicyTag.MULTI_PERIOD,
)
@RouteConstructorRegistry.register("esdp")
class ExactSDPPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Adapter bridging WSmart-Route simulation loop to the Exact SDP Engine.
    Now optimized for multi-period rolling horizon re-optimization.
    """

    @classmethod
    def _config_class(cls) -> Type[ExactSDPConfig]:
        """Return the SDP configuration dataclass."""
        return ExactSDPConfig

    def _get_config_key(self) -> str:
        """Return the configuration key."""
        return "esdp"

    def _run_multi_period_solver(
        self,
        problem: ProblemContext,
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """
        Execute the Exact Stochastic Dynamic Programming (ESDP) solver logic.

        This method solves the Recursive Bellman Equation over the $T$-period
        horizon by exhaustively traversing a discretized state-space representing
        bin inventory levels. It computes an optimal lookup table (value
        function) that prescribes the subset of nodes to collect at each
        stage to maximize expected long-term net profit.

        Because the state-space grows exponentially ($L^N$), this solver is
        strictly intended as an exact benchmark for small instances.

        Args:
            problem: The current ProblemContext containing the state, ScenarioTree,
                and problem constants (capacity, revenue, etc.).
            multi_day_ctx: Optional context for spanning multiple rolling days.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
                - today_solution: Standardized solution context for Day 0.
                - full_plan: Collection plan (nested list by day and vehicle).
                - stats: Engine statistics, state tuples, and optimal subset info.
        """
        cfg_dict = asdict(self.config) if self.config else {}
        params = SDPParams.from_config(cfg_dict)
        sub_dist_matrix = problem.distance_matrix
        sub_wastes = problem.wastes
        mandatory_nodes = problem.mandatory

        N = sub_dist_matrix.shape[0]
        # In multi-period terms, we treat 'day' as our current stage
        day_idx = multi_day_ctx.day_index if multi_day_ctx else 0
        day = day_idx + 1  # Framework uses 0-based, ESDP uses 1-based

        # 1. State Mapping
        state_lst = []
        for i in range(1, N):
            w = sub_wastes.get(i, 0.0)
            lvl = int(round((w / max(1.0, problem.capacity)) * (params.discrete_levels - 1)))
            lvl = max(0, min(params.discrete_levels - 1, lvl))
            state_lst.append(lvl)
        state_tuple = tuple(state_lst)

        # 2. Engine Initialization (Cached)
        cache_key = (N, params.num_days, params.discrete_levels, params.max_fill_rate)
        if cache_key not in _SDP_CACHE:
            engine = ExactSDPEngine(params, sub_dist_matrix, problem.capacity)
            engine.solve()
            _SDP_CACHE[cache_key] = engine

        engine = _SDP_CACHE[cache_key]

        # 3. Get Optimal Day 0 Action
        optimal_subset = list(engine.get_optimal_action(day, state_tuple))
        for mn in mandatory_nodes:
            if mn not in optimal_subset:
                optimal_subset.append(mn)

        # 4. Extract Tour (Simple TSP over subset)
        best_cost = float("inf")
        best_tour = [0, 0]
        for perm in itertools.permutations(optimal_subset):
            tour = [0] + list(perm) + [0]
            cost = sum(sub_dist_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))
            if cost < best_cost:
                best_cost = cost
                best_tour = tour

        # 5. Build T-period plan snapshot (optional/recursive)
        full_plan: List[List[List[int]]] = [[] for _ in range(params.num_days)]
        today_route = best_tour[1:-1] if len(best_tour) > 2 else []
        full_plan[0] = [today_route]

        sol_ctx = SolutionContext.from_problem(problem, today_route)

        return sol_ctx, full_plan, {"state": state_tuple, "subset": optimal_subset}

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """Legacy fallback."""
        return [], 0.0, 0.0
