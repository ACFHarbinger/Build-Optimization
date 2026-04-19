from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import PSOConfig
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import PSOParams
from .solver import PSOSolver


@RouteConstructorRegistry.register("pso")
class PSOPolicyAdapter(BaseMultiPeriodRoutingPolicy):
    """
    Policy adapter for Particle Swarm Optimization with velocity momentum.

    **TRUE PSO IMPLEMENTATION** (Kennedy & Eberhart 1995).
    Replaces the Sine Cosine Algorithm (SCA) which is mathematically
    equivalent to PSO without velocity momentum and with expensive
    trigonometric operations.

    Mathematical Superiority over SCA:
        PSO: v' = w*v + c₁*r₁*(pbest - x) + c₂*r₂*(gbest - x)
        SCA: x' = x + r₁·sin(r₂)·|r₃·gbest - x|

        Where SCA's sin(r₂) is just a random weight in [-1,1] with
        expensive transcendental computation and no periodicity exploitation.
    """

    def __init__(self, config: Optional[Union[PSOConfig, Dict[str, Any]]] = None):
        """
        Initialize PSO policy adapter.

        Args:
            config: Configuration parameters matching PSOParams fields.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return PSOConfig

    def _get_config_key(self) -> str:
        return "pso"

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
        guided toward the best known positions in the search-space ("gbest"),
        which are updated as better positions are found by other particles.

        Args:
            problem: The current ProblemContext containing the state, inventory,
                and distance matrices.
            multi_day_ctx: Optional context for inter-day state propagation.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
                - today_solution: Standardized solution context for Day 0.
                - full_plan: Collection plan (nested list by day and vehicle).
                - stats: Execution statistics and PSO solver metadata.
        """
        values = asdict(self.config) if self.config else {}

        params = PSOParams(
            pop_size=values.get("pop_size", 30),
            inertia_weight_start=values.get("inertia_weight_start", 0.9),
            inertia_weight_end=values.get("inertia_weight_end", 0.4),
            cognitive_coef=values.get("cognitive_coef", 2.0),
            social_coef=values.get("social_coef", 2.0),
            position_min=values.get("position_min", -1.0),
            position_max=values.get("position_max", 1.0),
            velocity_max=values.get("velocity_max", 0.5),
            max_iterations=values.get("max_iterations", 500),
            time_limit=values.get("time_limit", 60.0),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        solver = PSOSolver(
            dist_matrix=problem.distance_matrix,
            wastes=problem.wastes,
            capacity=problem.capacity,
            R=problem.revenue_per_kg,
            C=problem.cost_per_km,
            params=params,
            mandatory_nodes=problem.mandatory,
        )

        routes, profit, cost = solver.solve()

        # Wrap plan
        full_plan: List[List[List[int]]] = [[] for _ in range(self.horizon + 1)]
        full_plan[0] = routes

        # Extract primary route
        today_route = routes[0] if routes else []
        sol_ctx = SolutionContext.from_problem(problem, today_route)

        return sol_ctx, full_plan, {"cost": cost, "profit": profit}

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """Legacy fallback."""
        return [], 0.0, 0.0

    def get_name(self) -> str:
        """Return policy name."""
        return "Particle Swarm Optimization (PSO)"

    def get_acronym(self) -> str:
        """Return policy acronym."""
        return "PSO"
