"""
Evolutionary operators for Hybrid Genetic Search (HGS).

This module contains the crossover, evaluation, and fitness calculation
logic for maintaining and improving the HGS population.
"""

from typing import List, Optional

import numpy as np

from .individual import Individual
from .split import LinearSplit


def _extract_edges(individual: Individual) -> set:
    """
    Extract all undirected edges from an individual's routes.

    An edge {A, B} exists if node B immediately follows node A in a route
    (or vice-versa), including edges from/to the depot (node 0).
    Using frozenset ensures edges are undirected (Fix 8).

    Args:
        individual: Individual whose edges to extract.

    Returns:
        Set of undirected edges (frozensets of node pairs).
    """
    edges = set()
    for route in individual.routes:
        if not route:
            continue
        # Edge from depot to first node
        edges.add(frozenset({0, route[0]}))
        # Edges within route
        for i in range(len(route) - 1):
            edges.add(frozenset({route[i], route[i + 1]}))
        # Edge from last node back to depot
        edges.add(frozenset({route[-1], 0}))
    return edges


def _compute_broken_pairs_distance(ind1: Individual, ind2: Individual) -> float:
    """
    Compute the broken pairs distance (edge-based diversity) between two individuals.

    Following Vidal (2022), the distance is defined as:
    Δ(A,B) = 1 - |E(A) ∩ E(B)| / max(|E(A)|, |E(B)|)

    Args:
        ind1: First individual.
        ind2: Second individual.

    Returns:
        Broken pairs distance between 0.0 and 1.0.
    """
    edges1 = _extract_edges(ind1)
    edges2 = _extract_edges(ind2)

    if not edges1 and not edges2:
        return 0.0

    if not edges1 or not edges2:
        return 1.0

    intersection = len(edges1 & edges2)
    max_edges = max(len(edges1), len(edges2))

    return 1.0 - (intersection / max_edges)


def update_biased_fitness(
    population: List[Individual],
    nb_elite: int,
    neighbor_size: int = 5,
    distance_cache: Optional[dict] = None,
):
    """
    Update biased fitness based on profit rank and diversity rank.
    Follows Vidal et al. (2022) HGS-CVRP implementation with parameterless
    diversity weighting.

    Args:
        population: List of individuals to update (single subpopulation).
        nb_elite: Number of elite individuals to protect.
        neighbor_size: Number of nearest neighbors (nbClose) to consider for diversity.
        distance_cache: Optional cache for pairwise diversity distances (Fix 9).
    """
    if not population:
        return
    pop_size = len(population)
    if pop_size == 0:
        return

    # Rank by penalized objective: profit minus the penalty term for capacity violations.
    # For feasible individuals capacity_violation == 0 so this equals profit_score.
    # For infeasible individuals this penalises solutions with greater violations,
    # matching Vidal (2022) which ranks by solution quality including penalties.
    # Fix 10: Use the named property penalized_profit for sorting.
    population.sort(key=lambda x: x.penalized_profit, reverse=True)
    for i, ind in enumerate(population):
        ind.rank_profit = i + 1

    # Diversity: Average broken pairs distance to nbClose closest individuals
    # Based on edge similarity (topological diversity)
    # Fix 11: Compute pairwise distance matrix using symmetry (O(P^2) -> O(P^2/2)).
    n = len(population)
    dist_matrix_local = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            d = 0.0
            if distance_cache is not None:
                key = (id(population[i]), id(population[j]))
                if key not in distance_cache:
                    d = _compute_broken_pairs_distance(population[i], population[j])
                    distance_cache[key] = d
                    distance_cache[(id(population[j]), id(population[i]))] = d
                else:
                    d = distance_cache[key]
            else:
                d = _compute_broken_pairs_distance(population[i], population[j])

            dist_matrix_local[i][j] = d
            dist_matrix_local[j][i] = d

    for i, ind in enumerate(population):
        dists = sorted(dist_matrix_local[i][j] for j in range(n) if j != i)
        if dists:
            n_neighbors = min(neighbor_size, len(dists))
            ind.dist_to_parents = float(np.mean(dists[:n_neighbors]))
        else:
            ind.dist_to_parents = 0.0

    # Rank by Diversity (rank 1 = most diverse = best diversity contribution).
    # Since fitness is minimized in tournament selection, rank 1 gives the lowest
    # diversity term and thus the most diverse individual is preferred. This matches
    # Vidal (2022) Eq. (1): f(S) = rank_cost(S) + (1 - n_elite/|P|) * rank_div(S).
    population.sort(key=lambda x: x.dist_to_parents, reverse=True)
    for i, ind in enumerate(population):
        ind.rank_diversity = i + 1

    # Parameterless Biased Fitness (Vidal 2022)
    # BF(I) = Rank_C(I) + (1 - N_elite/|Pop|) * Rank_D(I)
    # This automatically balances profit and diversity based on elite proportion
    diversity_weight = 1.0 - (nb_elite / pop_size)

    for ind in population:
        # Apply biased fitness formula
        ind.fitness = ind.rank_profit + diversity_weight * ind.rank_diversity


def evaluate(ind: Individual, split_manager: LinearSplit, penalty_capacity: float = 1.0):
    """
    Decode giant tour and calculate metrics, including feasibility and penalties.

    This function transforms the genotype (giant_tour) into the phenotype (routes)
    using the Split algorithm. For VRPP, the Split algorithm may skip unprofitable nodes,
    resulting in routes that contain a subset of the nodes in giant_tour.

    The giant_tour is preserved unchanged to maintain genetic material for crossover.
    Nodes not in routes are considered "unvisited" and remain available for future
    insertion by local search operators.

    Args:
        ind: Individual to evaluate.
        split_manager: Split algorithm manager.
        penalty_capacity: Penalty coefficient for capacity violations.
    """
    routes, profit = split_manager.split(ind.giant_tour)
    ind.routes = routes
    ind.profit_score = profit

    # Calculate cost/revenue and check feasibility
    rev = 0.0
    cost = 0.0
    total_capacity_violation = 0.0

    for r in routes:
        if not r:
            continue

        # Calculate route distance and revenue
        d = split_manager.dist_matrix[0, r[0]]
        route_load = split_manager.wastes[r[0]]

        for i in range(len(r) - 1):
            d += split_manager.dist_matrix[r[i], r[i + 1]]
            route_load += split_manager.wastes[r[i + 1]]

        d += split_manager.dist_matrix[r[-1], 0]
        cost += d * split_manager.C

        # Calculate revenue for all nodes in route
        for node in r:
            rev += split_manager.wastes[node] * split_manager.R

        # Check capacity constraint
        if route_load > split_manager.capacity:
            total_capacity_violation += route_load - split_manager.capacity

    ind.cost = cost
    ind.revenue = rev
    ind.capacity_violation = total_capacity_violation
    ind.is_feasible = total_capacity_violation == 0.0

    # Penalized cost includes penalty for capacity violations
    ind.penalized_cost = cost + penalty_capacity * total_capacity_violation
