"""
Genetic Transformation (GT) Perturbation Module.

A mutation operator that takes a current solution, references an elite
solution from a historical pool, and enforces the preservation of
common edges (substructures) while randomly rewiring the rest.

Algorithm:
1. Identify all edges present in both the current and elite solutions.
2. Lock (preserve) those common edges.
3. Remove all non-common nodes from the current solution.
4. Greedily reinsert non-common nodes using cheapest insertion.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.perturbation.genetic_transformation import (
    ...     genetic_transformation,
    ... )
    >>> routes = genetic_transformation(routes, elite, dist_matrix, wastes, capacity, rng)
"""

from random import Random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def genetic_transformation(
    routes: List[List[int]],
    elite_solution: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    rng: Optional[Random] = None,
) -> List[List[int]]:
    """
    Genetic transformation: preserve common edges, rewire the rest.

    Args:
        routes: Current solution (list of routes).
        elite_solution: Elite reference solution.
        dist_matrix: Distance matrix.
        wastes: Waste/demand look-up.
        capacity: Vehicle capacity.
        rng: Random number generator.

    Returns:
        Modified routes after transformation.
    """
    if rng is None:
        rng = Random()

    # Collect edges from both solutions
    current_edges = _extract_edges(routes)
    elite_edges = _extract_edges(elite_solution)
    common_edges = current_edges & elite_edges

    # Identify locked nodes (nodes that are part of common edges)
    locked_nodes: Set[int] = set()
    for u, v in common_edges:
        if u != 0:
            locked_nodes.add(u)
        if v != 0:
            locked_nodes.add(v)

    # Remove non-locked nodes
    removed: List[int] = []
    for route in routes:
        to_remove = [n for n in route if n not in locked_nodes]
        removed.extend(to_remove)
        for n in to_remove:
            route.remove(n)

    # Clean empty routes
    routes = [r for r in routes if r]

    # Shuffle removed nodes for diversity
    rng.shuffle(removed)

    # Greedy reinsertion
    loads = [sum(wastes.get(n, 0) for n in r) for r in routes]

    for node in removed:
        node_waste = wastes.get(node, 0)
        best_cost = float("inf")
        best_route = -1
        best_pos = -1

        for r_idx, route in enumerate(routes):
            if loads[r_idx] + node_waste > capacity:
                continue
            for pos in range(len(route) + 1):
                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos] if pos < len(route) else 0
                cost = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                if cost < best_cost:
                    best_cost = cost
                    best_route = r_idx
                    best_pos = pos

        if best_route >= 0:
            routes[best_route].insert(best_pos, node)
            loads[best_route] += node_waste
        else:
            routes.append([node])
            loads.append(node_waste)

    return routes


def genetic_transformation_profit(
    routes: List[List[int]],
    elite_solution: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    rng: Optional[Random] = None,
) -> List[List[int]]:
    """
    Profit-driven genetic transformation: preserve common edges, rewire with profit reinsertion.

    Args:
        routes: Current solution (list of routes).
        elite_solution: Elite reference solution.
        dist_matrix: Distance matrix.
        wastes: Waste/demand look-up.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        rng: Random number generator.

    Returns:
        Modified routes after transformation.
    """
    if rng is None:
        rng = Random()

    # Collect edges from both solutions
    current_edges = _extract_edges(routes)
    elite_edges = _extract_edges(elite_solution)
    common_edges = current_edges & elite_edges

    # Identify locked nodes (nodes that are part of common edges)
    locked_nodes: Set[int] = set()
    for u, v in common_edges:
        if u != 0:
            locked_nodes.add(u)
        if v != 0:
            locked_nodes.add(v)

    # Remove non-locked nodes
    removed: List[int] = []
    for route in routes:
        to_remove = [n for n in route if n not in locked_nodes]
        removed.extend(to_remove)
        for n in to_remove:
            route.remove(n)

    # Clean empty routes
    routes = [r for r in routes if r]

    # Shuffle removed nodes for diversity
    rng.shuffle(removed)

    # Profit-driven reinsertion
    loads = [sum(wastes.get(n, 0) for n in r) for r in routes]

    for node in removed:
        node_waste = wastes.get(node, 0)
        revenue = node_waste * R
        best_profit = -float("inf")
        best_route = -1
        best_pos = -1

        for r_idx, route in enumerate(routes):
            if loads[r_idx] + node_waste > capacity:
                continue
            for pos in range(len(route) + 1):
                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos] if pos < len(route) else 0
                cost_inc = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                profit = revenue - (cost_inc * C)

                if profit > best_profit:
                    best_profit = profit
                    best_route = r_idx
                    best_pos = pos

        if best_route >= 0 and best_profit > -1e-4:
            routes[best_route].insert(best_pos, node)
            loads[best_route] += node_waste
        else:
            # Check if starting a new route is profitable
            cost_new = dist_matrix[0, node] + dist_matrix[node, 0]
            profit_new = revenue - (cost_new * C)
            if profit_new > -1e-4:
                routes.append([node])
                loads.append(node_waste)
            # Else: node remains unvisited

    return routes


def _extract_edges(solution: List[List[int]]) -> Set[Tuple[int, int]]:
    """Extract all directed edges from a solution (including depot connections)."""
    edges: Set[Tuple[int, int]] = set()
    for route in solution:
        if not route:
            continue
        edges.add((0, route[0]))
        for i in range(len(route) - 1):
            edges.add((route[i], route[i + 1]))
        edges.add((route[-1], 0))
    return edges
