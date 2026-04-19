"""
Differential Evolution Mutation Module.

Implements DE/rand/1 and DE/best/1 mutation strategies adapted for VRP
permutation chromosomes.  Classical DE operates on real-valued vectors:

    v_i = x_{r1} + F * (x_{r2} - x_{r3})

For permutation spaces, the "vector difference" (x_{r2} - x_{r3}) is
interpreted as the set of edges present in x_{r2} but absent in x_{r3}.
The base vector x_{r1} (or the population best for DE/best/1) is then
patched by injecting those differential edges via an Order Crossover (OX)
step weighted by the scaling factor F.

Algorithm for DE/rand/1:
1. For target chromosome x_i, select 3 distinct donors r1, r2, r3 ≠ i.
2. Compute the differential edge set: edges(x_r2) ∖ edges(x_r3).
3. Extract a contiguous segment of x_r1 of length ceil(F * len(x_i)).
4. Fill the remainder using the differential edges, then fall back to x_r1
   ordering — exactly the OX1 strategy restricted to differential positions.

DE/best/1 replaces x_{r1} with the population-best chromosome.

References:
    Storn, R., & Price, K. (1997). Differential evolution — a simple and
    efficient heuristic for global optimisation over continuous spaces.
    Journal of Global Optimization, 11(4), 341–359.

    Onwubolu, G. C., & Davendra, D. (2009). Scheduling flow shops using
    differential evolution algorithm. European Journal of Operational
    Research, 171(2), 674–692.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.evolutionary_mutation.differential_evolution import (
    ...     de_rand_1_mutation,
    ... )
    >>> mutated_pop = de_rand_1_mutation(population, distance_matrix, capacity, wastes, F=0.5)
"""

import math
from random import Random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def de_rand_1_mutation(
    population: List[List[List[int]]],
    distance_matrix: np.ndarray,
    capacity: float,
    wastes: Union[Dict[int, float], np.ndarray],
    F: float = 0.5,
    rng: Optional[Random] = None,
) -> List[List[List[int]]]:
    """
    DE/rand/1 mutation for a population of CVRP solutions.

    For each individual in the population, generates a mutant using three
    randomly chosen distinct donors and an OX1-based permutation analogue
    of the DE/rand/1 update rule.

    Args:
        population: List of solutions, each a list of routes (List[List[int]]).
        distance_matrix: N x N distance matrix.
        capacity: Vehicle capacity constraint.
        wastes: Node demands (dict or 1-D array).
        F: Differential weight ∈ (0, 2].  Controls the fraction of the
            chromosome that is rebuilt from the differential information.
        rng: Random number generator.

    Returns:
        List[List[List[int]]]: New list of mutant solutions (one per individual).
    """
    if rng is None:
        rng = Random()

    if len(population) < 4:
        return [_copy_solution(s) for s in population]

    mutants: List[List[List[int]]] = []
    indices = list(range(len(population)))

    for i, target in enumerate(population):
        donors_idx = rng.sample([j for j in indices if j != i], 3)
        r1, r2, r3 = (population[d] for d in donors_idx)
        mutant = _de_mutate(target, r1, r2, r3, F, capacity, wastes, rng)
        mutants.append(mutant)

    return mutants


def de_best_1_mutation(
    population: List[List[List[int]]],
    distance_matrix: np.ndarray,
    capacity: float,
    wastes: Union[Dict[int, float], np.ndarray],
    F: float = 0.5,
    rng: Optional[Random] = None,
) -> List[List[List[int]]]:
    """
    DE/best/1 mutation for a population of CVRP solutions.

    Uses the population-best solution (lowest total distance) as the base
    vector instead of a randomly chosen donor.  DE/best/1 converges faster
    than DE/rand/1 but may sacrifice diversity; use with caution on
    multi-modal landscapes.

    Args:
        population: List of solutions, each a list of routes (List[List[int]]).
        distance_matrix: N x N distance matrix.
        capacity: Vehicle capacity constraint.
        wastes: Node demands (dict or 1-D array).
        F: Differential weight ∈ (0, 2].
        rng: Random number generator.

    Returns:
        List[List[List[int]]]: New list of mutant solutions.
    """
    if rng is None:
        rng = Random()

    if len(population) < 3:
        return [_copy_solution(s) for s in population]

    # Determine population best by total route cost
    best_idx = int(np.argmin([_total_cost(sol, distance_matrix) for sol in population]))
    best = population[best_idx]

    indices = list(range(len(population)))
    mutants: List[List[List[int]]] = []

    for i, target in enumerate(population):
        donors_idx = rng.sample([j for j in indices if j != i and j != best_idx], 2)
        r2, r3 = (population[d] for d in donors_idx)
        mutant = _de_mutate(target, best, r2, r3, F, capacity, wastes, rng)
        mutants.append(mutant)

    return mutants


# ---------------------------------------------------------------------------
# Core DE permutation mutation logic
# ---------------------------------------------------------------------------


def _de_mutate(
    target: List[List[int]],
    base: List[List[int]],
    donor2: List[List[int]],
    donor3: List[List[int]],
    F: float,
    capacity: float,
    wastes: Union[Dict[int, float], np.ndarray],
    rng: Random,
) -> List[List[int]]:
    """
    Core DE/*/1 mutation step adapted for permutation chromosomes.

    1. Encode all four solutions as flat chromosomes (depot = -1).
    2. Build the "differential node set": nodes whose position in donor2
       differs significantly from donor3 (rank-difference heuristic).
    3. Inject a fraction F of those differentials into the base chromosome
       using OX1 splicing.
    4. Decode and return the mutant routes.
    """
    target_chrom, _ = _encode(target)
    base_chrom, n_v = _encode(base)
    chrom2, _ = _encode(donor2)
    chrom3, _ = _encode(donor3)

    # Extract real nodes from each chromosome (exclude -1 depot markers)
    nodes_base = [g for g in base_chrom if g != -1]
    nodes_d2 = [g for g in chrom2 if g != -1]
    nodes_d3 = [g for g in chrom3 if g != -1]
    nodes_target = [g for g in target_chrom if g != -1]

    if len(nodes_base) < 3:
        return _copy_solution(target)

    # Build rank maps for donors
    rank2 = {node: i for i, node in enumerate(nodes_d2)}
    rank3 = {node: i for i, node in enumerate(nodes_d3)}

    # Differential: nodes whose rank difference across donors is large
    all_nodes = list(dict.fromkeys(nodes_base + nodes_target))
    diffs: List[Tuple[int, float]] = []
    for node in all_nodes:
        r2 = rank2.get(node, len(nodes_d2))
        r3 = rank3.get(node, len(nodes_d3))
        diffs.append((node, abs(r2 - r3)))

    diffs.sort(key=lambda x: x[1], reverse=True)

    # Select top-F fraction of differential nodes
    n_diff = max(1, math.ceil(F * len(all_nodes)))
    diff_nodes = {node for node, _ in diffs[:n_diff]}

    # OX1 splice: keep a random segment of base_chrom, fill from differential ordering
    if len(nodes_base) < 2:
        return _copy_solution(base)

    c1, c2 = sorted(rng.sample(range(len(nodes_base)), 2))
    child: List[Optional[int]] = [None] * len(nodes_base)
    child[c1:c2] = nodes_base[c1:c2]
    inherited = set(nodes_base[c1:c2])

    # Fill positions from differential nodes first, then base ordering
    fill_order = [n for n in diff_nodes if n not in inherited] + [
        n for n in nodes_base if n not in inherited and n not in diff_nodes
    ]

    f_idx = 0
    for pos in range(len(nodes_base)):
        t_pos = (c2 + pos) % len(nodes_base)
        if child[t_pos] is None and f_idx < len(fill_order):
            child[t_pos] = fill_order[f_idx]
            f_idx += 1

    # Any remaining None slots get filled with leftover base nodes
    leftover = [n for n in nodes_base if n not in set(x for x in child if x is not None)]
    for pos in range(len(child)):
        if child[pos] is None and leftover:
            child[pos] = leftover.pop(0)

    clean_child = [x for x in child if x is not None]

    # Re-split into routes using the same vehicle count as the base solution
    return _split_into_routes(clean_child, n_v, capacity, wastes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode(routes: List[List[int]]) -> Tuple[List[int], int]:
    chromosome: List[int] = []
    for i, route in enumerate(routes):
        chromosome.extend(route)
        if i < len(routes) - 1:
            chromosome.append(-1)
    return chromosome, len(routes)


def _copy_solution(sol: List[List[int]]) -> List[List[int]]:
    return [r[:] for r in sol]


def _get_demand(wastes: Union[Dict[int, float], np.ndarray], node: int) -> float:
    if isinstance(wastes, dict):
        return wastes.get(node, 0.0)
    return float(wastes[node]) if node < len(wastes) else 0.0


def _total_cost(routes: List[List[int]], dist: np.ndarray) -> float:
    total = 0.0
    for route in routes:
        if not route:
            continue
        total += float(dist[0, route[0]])
        for i in range(len(route) - 1):
            total += float(dist[route[i], route[i + 1]])
        total += float(dist[route[-1], 0])
    return total


def _split_into_routes(
    nodes: List[int],
    n_vehicles: int,
    capacity: float,
    wastes: Union[Dict[int, float], np.ndarray],
) -> List[List[int]]:
    """
    Greedily assign the flat node sequence to routes respecting capacity.
    If the flat sequence contains more nodes than fit in n_vehicles routes,
    additional routes are opened.
    """
    routes: List[List[int]] = []
    current: List[int] = []
    load = 0.0
    for node in nodes:
        demand = _get_demand(wastes, node)
        if load + demand > capacity + 1e-6 and current:
            routes.append(current)
            current = [node]
            load = demand
        else:
            current.append(node)
            load += demand
    if current:
        routes.append(current)
    return routes
