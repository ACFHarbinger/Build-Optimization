"""
Evolutionary Perturbation Module.

Implements a micro-evolutionary algorithm (rapid GA) applied to a spatial
cluster of routes. Features dynamic route load-balancing via Dummy Depot
Injection and active unvisited node harvesting for VRPP profit maximization.

Algorithm:
1. Select a cluster of routes and flatten them into a "Giant Tour".
2. Inject Dummy Depots (-1) to represent vehicle boundaries.
3. (VRPP Only): Harvest high-profit unvisited nodes and swap them in.
4. Evolve via True OX1 Crossover and Swap Mutation.
5. Decode the optimal split and replace the original cluster.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.perturbation.evolutionary import (
    ...     evolutionary_perturbation,
    ... )
    >>> improved = evolutionary_perturbation(ls, target_routes=[0, 1], pop_size=10, n_gen=5)
"""

from random import Random
from typing import Dict, List, Optional, Union

import numpy as np


def evolutionary_perturbation(
    routes: List[List[int]],
    distance_matrix: np.ndarray,
    capacity: float,
    wastes: Union[Dict[int, float], np.ndarray],
    target_routes: List[List[int]],
    pop_size: int = 10,
    n_generations: int = 5,
    rng: Optional[Random] = None,
) -> List[List[int]]:
    """
    Micro-evolutionary perturbation for standard CVRP (Cost Minimization).

    Args:
        routes: Full global solution (list of all routes).
        distance_matrix: N x N distance matrix.
        capacity: Vehicle capacity constraint.
        wastes: Node demands.
        target_routes: The specific routes to be optimized/clustered.
        pop_size: GA population size.
        n_generations: GA generations.
        rng: Random generator.

    Returns:
        List[List[int]]: The updated full global solution.
    """
    if rng is None:
        rng = Random()

    if not routes or not target_routes:
        return routes

    # 1. Map target routes to their indices in the global solution
    target_indices = _map_target_to_global(routes, target_routes)
    if not target_indices:
        return routes

    # 2. Extract cluster nodes and build baseline chromosome
    cluster_nodes: List[int] = []
    for tr in target_routes:
        cluster_nodes.extend(tr)

    if len(cluster_nodes) < 3:
        return routes

    # Build chromosome: [node, node, -1 (depot), node...]
    base_chromosome = list(cluster_nodes)
    for _ in range(len(target_indices) - 1):
        base_chromosome.append(-1)

    baseline_cost = _eval_cvrp_chromosome(base_chromosome, distance_matrix, capacity, wastes)

    # 3. Generate Population
    population: List[List[int]] = [list(base_chromosome)]
    for _ in range(pop_size - 1):
        individual = list(base_chromosome)
        rng.shuffle(individual)
        population.append(individual)

    # 4. Evolve
    for _ in range(n_generations):
        fitnesses = [_eval_cvrp_chromosome(seq, distance_matrix, capacity, wastes) for seq in population]
        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i])
        survivors = [population[i] for i in ranked[: max(2, pop_size // 2)]]

        population = list(survivors)
        while len(population) < pop_size:
            p1 = rng.choice(survivors)
            p2 = rng.choice(survivors)
            child = _order_crossover(p1, p2, rng)
            _mutate_swap(child, rng, prob=0.3)
            population.append(child)

    # 5. Final Evaluation & Integration
    final_fitnesses = [_eval_cvrp_chromosome(seq, distance_matrix, capacity, wastes) for seq in population]
    best_idx = int(np.argmin(final_fitnesses))
    best_seq = population[best_idx]

    if final_fitnesses[best_idx] < baseline_cost - 1e-4:
        return _apply_to_solution(routes, target_indices, best_seq)

    return routes


def evolutionary_perturbation_profit(
    routes: List[List[int]],
    distance_matrix: np.ndarray,
    capacity: float,
    wastes: Union[Dict[int, float], np.ndarray],
    revenue: float,
    cost_unit: float,
    target_routes: List[List[int]],
    pop_size: int = 10,
    n_generations: int = 5,
    rng: Optional[Random] = None,
) -> List[List[int]]:
    """
    Micro-evolutionary perturbation for VRPP (Profit Maximization).
    """
    if rng is None:
        rng = Random()

    if not routes or not target_routes:
        return routes

    # 1. Map target routes to global indices
    target_indices = _map_target_to_global(routes, target_routes)
    if not target_indices:
        return routes

    cluster_nodes: List[int] = []
    for tr in target_routes:
        cluster_nodes.extend(tr)

    if len(cluster_nodes) < 3:
        return routes

    # 2. Baseline baseline check
    base_chromosome_original = list(cluster_nodes)
    for _ in range(len(target_indices) - 1):
        base_chromosome_original.append(-1)

    baseline_profit = _eval_vrpp_chromosome(
        base_chromosome_original, distance_matrix, capacity, wastes, revenue, cost_unit
    )

    # 3. Profit-Aware Harvesting: Swap low-profit visited for high-profit unvisited
    all_visited = {n for r in routes for n in r}
    n_nodes = distance_matrix.shape[0]
    unvisited = [n for n in range(1, n_nodes) if n not in all_visited]
    if unvisited and cluster_nodes:
        # Swap logic (up to 15% of cluster)
        num_to_swap = max(1, min(len(unvisited), len(cluster_nodes) // 6))
        unvisited_sorted = sorted(unvisited, key=lambda n: _get_waste_val(wastes, n) * revenue, reverse=True)
        cluster_sorted = sorted(cluster_nodes, key=lambda n: _get_waste_val(wastes, n) * revenue)

        for i in range(num_to_swap):
            idx_to_replace = cluster_nodes.index(cluster_sorted[i])
            cluster_nodes[idx_to_replace] = unvisited_sorted[i]

    # 4. Chromosome Generation
    base_chromosome = list(cluster_nodes)
    for _ in range(len(target_indices) - 1):
        base_chromosome.append(-1)

    population: List[List[int]] = [list(base_chromosome)]
    for _ in range(pop_size - 1):
        individual = list(base_chromosome)
        rng.shuffle(individual)
        population.append(individual)

    # 5. Evolve
    for _ in range(n_generations):
        # Fitness is negative profit for rank-sorting
        fitnesses = [
            -_eval_vrpp_chromosome(seq, distance_matrix, capacity, wastes, revenue, cost_unit) for seq in population
        ]
        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i])
        survivors = [population[i] for i in ranked[: max(2, pop_size // 2)]]

        population = list(survivors)
        while len(population) < pop_size:
            p1 = rng.choice(survivors)
            p2 = rng.choice(survivors)
            child = _order_crossover(p1, p2, rng)
            _mutate_swap(child, rng, prob=0.3)
            population.append(child)

    # 6. Final Evaluation
    final_fitnesses = [
        _eval_vrpp_chromosome(seq, distance_matrix, capacity, wastes, revenue, cost_unit) for seq in population
    ]
    best_idx = int(np.argmax(final_fitnesses))
    best_seq = population[best_idx]

    if final_fitnesses[best_idx] > baseline_profit + 1e-4:
        return _apply_to_solution(routes, target_indices, best_seq)

    return routes


# --- Evaluators & Decoder ---


def _decode_chromosome(seq: List[int]) -> List[List[int]]:
    routes: List[List[int]] = []
    current_route: List[int] = []
    for gene in seq:
        if gene == -1:
            routes.append(current_route)
            current_route = []
        else:
            current_route.append(gene)
    routes.append(current_route)
    return routes


def _eval_cvrp_chromosome(
    seq: List[int], d: np.ndarray, cap: float, wastes: Union[Dict[int, float], np.ndarray]
) -> float:
    routes = _decode_chromosome(seq)
    total_cost = 0.0
    penalty = 0.0
    for r in routes:
        if not r:
            continue
        total_cost += _sequence_cost(d, r)
        load = sum(_get_waste_val(wastes, n) for n in r)
        if load > cap + 1e-6:
            penalty += (load - cap) * 10000.0
    return total_cost + penalty


def _eval_vrpp_chromosome(
    seq: List[int], d: np.ndarray, cap: float, wastes: Union[Dict[int, float], np.ndarray], R: float, C: float
) -> float:
    routes = _decode_chromosome(seq)
    total_profit = 0.0
    penalty = 0.0
    for r in routes:
        if not r:
            continue
        total_profit += _sequence_profit(r, d, wastes, R, C)
        load = sum(_get_waste_val(wastes, n) for n in r)
        if load > cap + 1e-6:
            penalty += (load - cap) * 10000.0
    return total_profit - penalty


# --- Core Logic ---


def _map_target_to_global(routes: List[List[int]], targets: List[List[int]]) -> List[int]:
    """Identify indices of provided sub-routes in the global solution."""
    indices = []
    temp_full = [r[:] for r in routes]
    for tr in targets:
        try:
            idx = temp_full.index(tr)
            indices.append(idx)
            temp_full[idx] = [-99]  # Mask to handle identical routes
        except ValueError:
            continue
    return indices


def _apply_to_solution(routes: List[List[int]], indices: List[int], best_seq: List[int]) -> List[List[int]]:
    """Integrates optimized sub-routes back into the full solution."""
    new_solution = [r[:] for r in routes]
    new_parts = _decode_chromosome(best_seq)
    for i, idx in enumerate(indices):
        new_solution[idx] = new_parts[i]
    return [r for r in new_solution if r]  # Clean empty routes


# --- Recombination & Mutation ---


def _order_crossover(p1: List[int], p2: List[int], rng: Random) -> List[int]:
    n = len(p1)
    if n < 2:
        return list(p1)

    def _tag(s):
        res, d_id = [], -1
        for x in s:
            if x == -1:
                res.append(d_id)
                d_id -= 1
            else:
                res.append(x)
        return res

    p1_t, p2_t = _tag(p1), _tag(p2)
    c1, c2 = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[c1:c2] = p1_t[c1:c2]
    inherited = set(p1_t[c1:c2])
    fill = [g for g in p2_t if g not in inherited]

    f_idx = 0
    for i in range(n):
        t_idx = (c2 + i) % n
        if child[t_idx] is None:
            child[t_idx] = fill[f_idx]
            f_idx += 1
    return [-1 if x < 0 else x for x in child]  # type: ignore[misc,operator]


def _mutate_swap(seq: List[int], rng: Random, prob: float) -> None:
    if len(seq) > 1 and rng.random() < prob:
        i, j = rng.sample(range(len(seq)), 2)
        seq[i], seq[j] = seq[j], seq[i]


# --- Helpers ---


def _get_waste_val(wastes: Union[Dict[int, float], np.ndarray], node: int) -> float:
    if isinstance(wastes, dict):
        return wastes.get(node, 0.0)
    return float(wastes[node]) if node < len(wastes) else 0.0


def _sequence_cost(d: np.ndarray, seq: List[int]) -> float:
    cost = d[0, seq[0]]
    for i in range(len(seq) - 1):
        cost += d[seq[i], seq[i + 1]]
    return float(cost + d[seq[-1], 0])


def _sequence_profit(
    seq: List[int], d: np.ndarray, wastes: Union[Dict[int, float], np.ndarray], R: float, C: float
) -> float:
    dist = d[0, seq[0]]
    rev = _get_waste_val(wastes, seq[0]) * R
    for i in range(len(seq) - 1):
        dist += d[seq[i], seq[i + 1]]
        rev += _get_waste_val(wastes, seq[i + 1]) * R
    return float(rev - (dist + d[seq[-1], 0]) * C)
