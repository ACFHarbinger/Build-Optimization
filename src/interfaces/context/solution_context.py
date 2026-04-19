"""
SolutionContext — Current-day routing solution.

Stores the routes actually executed today (one per vehicle) together with
per-vehicle profit and cost breakdowns and aggregate totals.
The multi-day plan beyond today is stored in MultiDayContext.full_plan_snapshot
and is NOT duplicated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .problem_context import ProblemContext


@dataclass(frozen=True)
class SolutionContext:
    """
    Today's routing solution, as returned by _run_multi_period_solver.

    Attributes:
        routes:        List of routes for today. Each route is a List[int] of
                       node IDs for one vehicle (depot excluded; e.g. [3, 7, 2]).
                       Empty list [] means that vehicle is idle.
                       Typically len(routes) == n_vehicles or 1 for single-vehicle.
        profits:       Per-route daily profit (r_w·Σw_i − c_km·dist).
                       len(profits) == len(routes).
        costs:         Per-route travel cost (c_km · dist only, no revenue).
                       len(costs) == len(routes).
        total_profit:  Sum of profits across all today's routes.
        total_cost:    Sum of costs across all today's routes.
        metadata:      Optional solver-specific telemetry (e.g. MIP gap,
                       nodes explored, LR iterations).
    """

    routes: List[List[int]]
    profits: List[float]
    costs: List[float]
    total_profit: float
    total_cost: float
    metadata: dict = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "SolutionContext":
        """Return an empty (no-visit) solution."""
        return cls(routes=[], profits=[], costs=[], total_profit=0.0, total_cost=0.0)

    @classmethod
    def from_single_route(
        cls,
        route: List[int],
        profit: float,
        cost: float,
        metadata: Optional[dict] = None,
    ) -> "SolutionContext":
        """Convenience constructor for single-vehicle results."""
        return cls(
            routes=[route],
            profits=[profit],
            costs=[cost],
            total_profit=profit,
            total_cost=cost,
            metadata=metadata or {},
        )

    @classmethod
    def from_problem(
        cls,
        problem: ProblemContext,
        route: List[int],
        metadata: Optional[dict] = None,
    ) -> "SolutionContext":
        """
        Calculate profit and cost for a single route from the problem context.

        Args:
            problem:  The ProblemContext containing distance matrix and weights.
            route:    The node sequence visited (1-based, depot excluded).
            metadata: Optional telemetry.
        """
        revenue = sum(problem.wastes.get(node, 0.0) for node in route) * problem.revenue_per_kg
        dist = 0.0
        if route:
            dist += problem.distance_matrix[0, route[0]]
            for i in range(len(route) - 1):
                dist += problem.distance_matrix[route[i], route[i + 1]]
            dist += problem.distance_matrix[route[-1], 0]
        cost = dist * problem.cost_per_km
        profit = revenue - cost
        return cls.from_single_route(route, profit, cost, metadata)

    @classmethod
    def from_multi_route(
        cls,
        problem: ProblemContext,
        routes: List[List[int]],
        metadata: Optional[dict] = None,
    ) -> "SolutionContext":
        """
        Calculate aggregate and per-route profits/costs for multiple vehicles.

        Args:
            problem:  The ProblemContext.
            routes:   List of route sequences (one per vehicle).
            metadata: Optional telemetry.
        """
        all_profits = []
        all_costs = []
        total_profit = 0.0
        total_cost = 0.0

        for route in routes:
            revenue = sum(problem.wastes.get(node, 0.0) for node in route) * problem.revenue_per_kg
            dist = 0.0
            if route:
                dist += problem.distance_matrix[0, route[0]]
                for i in range(len(route) - 1):
                    dist += problem.distance_matrix[route[i], route[i + 1]]
                dist += problem.distance_matrix[route[-1], 0]
            cost = dist * problem.cost_per_km
            profit = revenue - cost

            all_profits.append(profit)
            all_costs.append(cost)
            total_profit += profit
            total_cost += cost

        return cls(
            routes=routes,
            profits=all_profits,
            costs=all_costs,
            total_profit=total_profit,
            total_cost=total_cost,
            metadata=metadata or {},
        )

    def to_flat_tour(self) -> List[int]:
        """
        Flatten all routes into a single depot-delimited tour.

        Produces [0, r1_n1, r1_n2, ..., 0, r2_n1, ...., 0] suitable for
        passing back to the simulator's single-tour interface.
        """
        flat = []
        for route in self.routes:
            if route:
                flat.append(0)
                flat.extend(route)
        flat.append(0)
        return flat if flat else [0, 0]
