"""
Solution representation for GIHH.

This module defines the Solution class used by the GIHH algorithm.
"""

from typing import Dict, List

import numpy as np


class Solution:
    """
    Solution representation for Vehicle Routing Problem with Profits.

    Attributes:
        routes: List of routes, where each route is a list of node indices.
        dist_matrix: Distance matrix.
        wastes: Dictionary of node wastes.
        capacity: Vehicle capacity.
        revenue: Revenue multiplier.
        cost_unit: Cost multiplier.
        profit: Total profit (revenue - cost).
        cost: Total routing cost.
        revenue_total: Total revenue from collected waste.
    """

    def __init__(
        self,
        routes: List[List[int]],
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
    ):
        """
        Initialize solution.

        Args:
            routes: List of routes.
            dist_matrix: Distance matrix.
            wastes: Waste dictionary.
            capacity: Vehicle capacity.
            revenue: Revenue per unit.
            cost_unit: Cost per distance unit.
        """
        self.routes = routes
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.revenue = revenue
        self.cost_unit = cost_unit

        self.profit = 0.0
        self.cost = 0.0
        self.revenue_total = 0.0

        self.evaluate()

    def evaluate(self) -> None:
        """Evaluate the solution and update profit, cost, and revenue."""
        total_distance = 0.0
        total_waste = 0.0

        for route in self.routes:
            if len(route) == 0:
                continue

            # Calculate route distance: depot -> route -> depot
            route_distance = self.dist_matrix[0, route[0]]
            for i in range(len(route) - 1):
                route_distance += self.dist_matrix[route[i], route[i + 1]]
            route_distance += self.dist_matrix[route[-1], 0]

            total_distance += route_distance

            # Calculate route waste
            route_waste = sum(self.wastes.get(node, 0.0) for node in route)
            total_waste += route_waste

        self.cost = total_distance * self.cost_unit
        self.revenue_total = total_waste * self.revenue
        self.profit = self.revenue_total - self.cost

    def copy(self) -> "Solution":
        """Create a deep copy of the solution."""
        return Solution(
            routes=[route[:] for route in self.routes],
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            revenue=self.revenue,
            cost_unit=self.cost_unit,
        )

    def is_feasible(self) -> bool:
        """Check if solution respects capacity constraints."""
        for route in self.routes:
            route_load = sum(self.wastes.get(node, 0.0) for node in route)
            if route_load > self.capacity:
                return False
        return True

    def is_identical_to(self, other: "Solution") -> bool:
        """
        Check if this solution is structurally identical to another.
        Two solutions are identical if they have the exact same routes.
        """
        return self.routes == other.routes

    def dominates(self, other: "Solution") -> bool:
        """
        Check if this solution Pareto-dominates the other solution.
        Objectives: Maximize revenue_total, Minimize cost.
        """
        revenue_better_or_equal = self.revenue_total >= other.revenue_total
        cost_better_or_equal = self.cost <= other.cost

        revenue_strictly_better = self.revenue_total > other.revenue_total
        cost_strictly_better = self.cost < other.cost

        return (revenue_better_or_equal and cost_better_or_equal) and (revenue_strictly_better or cost_strictly_better)
