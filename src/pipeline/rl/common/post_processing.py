"""
Post-processing utilities.
Includes route decoding, efficiency calculation, and lightweight optimization.
"""

from typing import List

import torch


class EfficiencyOptimizer:
    """Optimizes routes post-generation."""

    def __init__(self, problem, **kwargs):
        """
        Initialize EfficiencyOptimizer.

        Args:
            problem: Description for problem.
            **kwargs: Description for **kwargs.
        """
        self.problem = problem

    def optimize(self, routes: List[torch.Tensor], **kwargs):
        """Refine routes using local search or heuristics."""
        # This matches the legacy EfficiencyOptimizer which might run 2-opt
        # We can link to our new vectorized policies here if needed
        return routes


def decode_routes(actions: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """Convert action sequences to clean route lists on CPU."""
    routes = []
    actions_cpu = actions.cpu()
    for action in actions_cpu:
        # Cut at 0s
        route = action[action != 0].tolist()
        routes.append(route)
    return routes


def calculate_efficiency(routes, dist_matrix, waste, capacity):
    """
    Compute kg/km efficiency metric.
    Efficiency = Total Waste Collected / Total Distance Traveled.
    """
    total_waste = 0.0
    total_dist = 0.0
    for route in routes:
        if not route:
            continue

        # 1. Calculate Waste
        route_tensor = torch.tensor(route, dtype=torch.long)

        # Gather waste
        # wastes needs to be accessible by index
        waste = waste[route_tensor].sum().item()
        total_waste += waste

        # 2. Calculate Distance
        # Depot -> First Node
        d = dist_matrix[0, route[0]].item()

        # Inter-node
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            d += dist_matrix[u, v].item()

        # Last Node -> Depot
        d += dist_matrix[route[-1], 0].item()

        total_dist += d

    if total_dist < 1e-6:
        return 0.0

    return total_waste / total_dist
