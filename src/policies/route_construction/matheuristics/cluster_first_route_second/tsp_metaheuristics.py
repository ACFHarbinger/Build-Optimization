"""
TSP Metaheuristic Solvers for Cluster-First Route-Second Algorithm.

This module provides metaheuristic algorithms for solving the Traveling Salesman Problem
in the routing phase of the VFJ algorithm.

Supported Metaheuristics (as evaluated in Sultana & Akhand, 2017):
- Particle Swarm Optimization (PSO) - FULLY IMPLEMENTED
- Ant Colony Optimization (ACO) - Placeholder
- Genetic Algorithm (GA) - Placeholder

The paper concludes that VFJ + PSO provides the best performance for CVRP benchmarks.

Reference:
    Sultana, T., Akhand, M. A. H., & Rahman, M. M. H. (2017).
    Section 4: "VFJ algorithm with PSO performs better than VFJ algorithm with
    ACO and VFJ algorithm with GA"
"""

import random
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def _calculate_tour_distance(tour: List[int], distance_matrix: np.ndarray) -> float:
    """Calculate total distance of a TSP tour."""
    total = 0.0
    for i in range(len(tour) - 1):
        total += distance_matrix[tour[i], tour[i + 1]]
    return total


def _two_opt_swap(tour: List[int], i: int, k: int) -> List[int]:
    """
    Perform 2-opt swap on tour segment between indices i and k.

    2-opt removes two edges and reconnects the path by reversing the segment.
    This is the fundamental local search operator for TSP.

    Args:
        tour: Current tour (includes depot at start and end)
        i: Start index of segment to reverse
        k: End index of segment to reverse

    Returns:
        New tour with reversed segment
    """
    new_tour = tour[:i] + tour[i : k + 1][::-1] + tour[k + 1 :]
    return new_tour


def _apply_swap_sequence(tour: List[int], swap_ops: List[Tuple[int, int]], max_swaps: int = 3) -> List[int]:
    """
    Apply a sequence of swap operations to move toward a target tour.

    This implements the discrete PSO velocity update mechanism by
    probabilistically applying swaps that move the current particle
    toward its personal best and the global best.

    Args:
        tour: Current tour
        swap_ops: List of (i, j) swap operations
        max_swaps: Maximum number of swaps to apply

    Returns:
        Modified tour
    """
    result = tour.copy()
    num_swaps = min(len(swap_ops), max_swaps)

    for i in range(num_swaps):
        idx1, idx2 = swap_ops[i]
        if 0 < idx1 < len(result) - 1 and 0 < idx2 < len(result) - 1:
            result[idx1], result[idx2] = result[idx2], result[idx1]

    return result


def _get_swap_sequence(source: List[int], target: List[int]) -> List[Tuple[int, int]]:
    """
    Compute sequence of swap operations to transform source tour into target tour.

    This is used to compute the "velocity" in discrete PSO by finding
    the swaps needed to move from current position to pbest/gbest.

    Args:
        source: Current tour
        target: Target tour

    Returns:
        List of (i, j) index pairs to swap
    """
    swaps = []
    temp = source.copy()

    for i in range(1, len(temp) - 1):  # Exclude depot at start and end
        if temp[i] != target[i]:
            # Find where target[i] is in temp
            for j in range(i + 1, len(temp) - 1):
                if temp[j] == target[i]:
                    swaps.append((i, j))
                    temp[i], temp[j] = temp[j], temp[i]
                    break

    return swaps


def _get_swap_operators(tour1: List[int], tour2: List[int]) -> List[Tuple[int, int]]:
    """Compute sequence of swap operations to transform tour1 into tour2."""
    swaps = []
    t1 = list(tour1)
    for i in range(len(t1)):
        if t1[i] != tour2[i]:
            for j in range(i + 1, len(t1)):
                if t1[j] == tour2[i]:
                    swaps.append((i, j))
                    t1[i], t1[j] = t1[j], t1[i]
                    break
    return swaps


def _apply_2opt_local_search(
    tour: List[int],
    distance_matrix: np.ndarray,
    n_customers: int,
    max_swaps: int = 10,
) -> List[int]:
    """Apply random 2-opt swaps to improve a tour."""
    best_tour = tour.copy()
    best_dist = _calculate_tour_distance(best_tour, distance_matrix)

    # Try a limited number of 2-opt swaps
    for _ in range(min(max_swaps, n_customers)):
        i = np.random.randint(1, n_customers)
        k = np.random.randint(i + 1, n_customers + 1)
        new_tour = _two_opt_swap(best_tour, i, k)
        new_dist = _calculate_tour_distance(new_tour, distance_matrix)

        if new_dist < best_dist:
            best_tour = new_tour
            best_dist = new_dist

    return best_tour


def find_route_pso(
    distance_matrix: np.ndarray,
    cluster: List[int],
    time_limit: float = 60.0,
    seed: int = 42,
    n_particles: int = 50,
    c1: float = 1.5,
    c2: float = 1.5,
) -> List[int]:
    """
    Solve TSP using Velocity Tentative PSO (VTPSO).

    Ref: Akhand et al. (2015); Sultana & Akhand (2017). VTPSO computes a
    tentative velocity and uses partial search to explore intermediate
    tours, selecting the best one. This methodology enables the mapping
    of continuous velocity vectors to discrete TSP topologies.

    Args:
        distance_matrix: Pre-computed all-pairs distance matrix.
        cluster: List of node indices to visit (including depot at index 0).
        time_limit: Maximum time in seconds.
        seed: Random seed.
        n_particles: Number of particles.
        c1: Cognitive coefficient.
        c2: Social coefficient.

    Returns:
        TSP tour as list of node indices [0, i1, i2, ..., in, 0].
    """
    np.random.seed(seed)
    random.seed(seed)
    start_time = time.time()

    if not cluster:
        return [0, 0]
    if len(cluster) <= 3:
        return list(cluster) + [0] if cluster[-1] != 0 else cluster

    customers = [node for node in cluster if node != 0]

    # Initialize particles
    particles = []
    for _ in range(n_particles):
        perm = np.random.permutation(customers).tolist()
        tour = [0] + perm + [0]
        particles.append(
            {"pos": tour, "pbest": tour.copy(), "pbest_dist": _calculate_tour_distance(tour, distance_matrix)}
        )

    gbest_tour = min(particles, key=lambda p: p["pbest_dist"])["pbest"].copy()
    gbest_dist = _calculate_tour_distance(gbest_tour, distance_matrix)

    while time.time() - start_time < time_limit:
        for p in particles:
            if time.time() - start_time >= time_limit:
                break

            # 1. Calculate Tentative Velocity
            v_pbest = _get_swap_operators(p["pos"], p["pbest"])
            v_gbest = _get_swap_operators(p["pos"], gbest_tour)

            # Stochastic selection of swaps
            v_pbest = [s for s in v_pbest if random.random() < 0.5]  # Simplified for VTPSO
            v_gbest = [s for s in v_gbest if random.random() < 0.5]

            tentative_velocity = v_pbest + v_gbest

            # 2. VTPSO Partial Search
            current_tour = list(p["pos"])
            best_inter_tour = list(current_tour)
            best_inter_dist = _calculate_tour_distance(best_inter_tour, distance_matrix)

            for idx1, idx2 in tentative_velocity:
                current_tour[idx1], current_tour[idx2] = current_tour[idx2], current_tour[idx1]
                d = _calculate_tour_distance(current_tour, distance_matrix)
                if d < best_inter_dist:
                    best_inter_dist = d
                    best_inter_tour = list(current_tour)

            p["pos"] = best_inter_tour

            # Update personal best
            if best_inter_dist < p["pbest_dist"]:
                p["pbest"] = list(p["pos"])
                p["pbest_dist"] = best_inter_dist

                # Update global best
                if best_inter_dist < gbest_dist:
                    gbest_tour = list(p["pbest"])
                    gbest_dist = best_inter_dist

    return gbest_tour


def find_route_aco(
    distance_matrix: np.ndarray,
    cluster: List[int],
    time_limit: float = 60.0,
    seed: int = 42,
    n_ants: Optional[int] = None,
    alpha: float = 1.0,
    beta: float = 3.0,
    rho: float = 0.1,
    q: float = 100.0,
) -> List[int]:
    """
    Solve TSP using Ant Colony Optimization (ACO).

    Args:
        distance_matrix: Pre-computed all-pairs distance matrix.
        cluster: List of node indices visit.
        time_limit: Maximum time.
        seed: Random seed.
        n_ants: Number of ants (defaults to len(cluster) per paper).
        alpha: Pheromone importance (Paper default: 1.0).
        beta: Heuristic importance (Paper default: 3.0).
        rho: Evaporation rate (Standard ACO: 0.1).
        q: Pheromone deposit amount (Standard ACO: 100.0).

    Note:
        The paper Sultana & Akhand (2017) specifies alpha=1 and beta=3 but is
        vague on evaporation and deposit scale. We adopt rho=0.1 and Q=100
        as robust defaults for the pheromone update Delta_tau = Q / L, where
        L is the tour distance.
    """
    np.random.seed(seed)
    random.seed(seed)
    start_time = time.time()

    if not cluster:
        return [0, 0]
    if len(cluster) <= 3:
        return list(cluster) + [0] if cluster[-1] != 0 else cluster

    n_nodes = len(cluster)
    if n_ants is None:
        n_ants = n_nodes  # Paper: "number of ants in ACO was equal to the number of cities"

    # Methodological Note on Pheromone Initialization:
    # Standard OR practice often initializes tau_0 = 1 / (N * L_nn), where L_nn is
    # the cost of a nearest-neighbor tour. We use a uniform initialization of 0.1
    # here as a robust baseline for simplicity and comparability.
    pheromones = np.ones((n_nodes, n_nodes)) * 0.1
    # Heuristic: 1/distance. Small epsilon to avoid div by zero.
    heuristics = 1.0 / (distance_matrix[cluster][:, cluster] + 1e-10)

    best_tour = list(cluster)
    if best_tour[-1] != 0:
        best_tour.append(0)
    best_dist = _calculate_tour_distance(best_tour, distance_matrix)

    while time.time() - start_time < time_limit:
        all_tours = []
        all_dists = []

        for _ in range(n_ants):
            if time.time() - start_time >= time_limit:
                break

            curr_idx = random.randint(0, n_nodes - 1)
            tour_indices = [curr_idx]
            unvisited = set(range(n_nodes))
            unvisited.remove(curr_idx)

            while unvisited:
                unvisited_arr = np.array(list(unvisited))

                # Vectorized probability calculation
                # (pheromones ** alpha) * (heuristics ** beta)
                p_values = (pheromones[curr_idx, unvisited_arr] ** alpha) * (
                    heuristics[curr_idx, unvisited_arr] ** beta
                )

                total = np.sum(p_values)
                if total == 0:
                    next_idx = random.choice(list(unvisited))
                else:
                    probs = p_values / total
                    next_idx = np.random.choice(unvisited_arr, p=probs)

                tour_indices.append(next_idx)
                unvisited.remove(next_idx)
                curr_idx = next_idx

            tour = [cluster[i] for i in tour_indices]
            # Ensure it ends at depot if it starts at depot, or just make it circular
            if 0 in tour:
                z_idx = tour.index(0)
                tour = tour[z_idx:] + tour[:z_idx]
            tour.append(tour[0])

            d = _calculate_tour_distance(tour, distance_matrix)
            all_tours.append(tour_indices)
            all_dists.append(d)

            if d < best_dist:
                best_dist = d
                best_tour = list(tour)

        pheromones *= 1.0 - rho
        for tour_indices, d in zip(all_tours, all_dists, strict=False):
            deposit = q / (d + 1e-10)
            for i in range(len(tour_indices)):
                n1, n2 = tour_indices[i], tour_indices[(i + 1) % len(tour_indices)]
                pheromones[n1, n2] += deposit
                pheromones[n2, n1] += deposit

    return best_tour


def _eer_crossover(p1: List[int], p2: List[int], customers: List[int]) -> List[int]:
    """Enhanced Edge Recombination (EER) crossover."""
    n_cust = len(customers)
    edge_table: Dict[int, Set[int]] = {node: set() for node in customers}
    for parent in [p1, p2]:
        for i in range(n_cust):
            edge_table[parent[i]].add(parent[(i - 1) % n_cust])
            edge_table[parent[i]].add(parent[(i + 1) % n_cust])

    child: List[int] = []
    curr = random.choice([p1[0], p2[0]])
    while len(child) < n_cust:
        child.append(curr)
        for neighbors in edge_table.values():
            neighbors.discard(curr)

        if len(child) == n_cust:
            break

        neighbors = list(edge_table[curr])  # type: ignore[assignment]
        if not neighbors:
            remaining = [node for node in customers if node not in child]
            curr = random.choice(remaining)
        else:
            min_neighbors = min(len(edge_table[n]) for n in neighbors)
            candidates = [n for n in neighbors if len(edge_table[n]) == min_neighbors]
            curr = random.choice(candidates)
    return child


def find_route_ga(
    distance_matrix: np.ndarray,
    cluster: List[int],
    time_limit: float = 60.0,
    seed: int = 42,
    pop_size: int = 50,
    mutation_rate: float = 0.2,
) -> List[int]:
    """
    Solve TSP using Genetic Algorithm (GA).

    Args:
        distance_matrix: Pre-computed all-pairs distance matrix.
        cluster: List of node indices.
        time_limit: Maximum time.
        seed: Random seed.
        pop_size: Population size.
        mutation_rate: Mutation probability.
    """
    np.random.seed(seed)
    random.seed(seed)
    start_time = time.time()

    if not cluster:
        return [0, 0]
    if len(cluster) <= 3:
        return list(cluster) + [0] if cluster[-1] != 0 else cluster

    customers = [node for node in cluster if node != 0]
    n_cust = len(customers)

    def _get_tour(perm):
        return [0] + perm + [0]

    def _get_dist(perm):
        return _calculate_tour_distance(_get_tour(perm), distance_matrix)

    # Initialize population
    population = []
    for _ in range(pop_size):
        perm = np.random.permutation(customers).tolist()
        population.append(perm)

    best_perm = min(population, key=_get_dist)
    best_dist = _get_dist(best_perm)

    while time.time() - start_time < time_limit:
        population.sort(key=_get_dist)
        new_pop = population[:2]  # Elitism

        while len(new_pop) < pop_size:
            if time.time() - start_time >= time_limit:
                break
            # Selection (Tournament)
            p1 = min(random.sample(population, min(5, len(population))), key=_get_dist)
            p2 = min(random.sample(population, min(5, len(population))), key=_get_dist)

            # Crossover (Enhanced Edge Recombination - EER)
            # Ref: Sultana & Akhand (2017) specify EER for adjacency preservation
            child = _eer_crossover(p1, p2, customers)

            # Mutation (Swap)
            if random.random() < mutation_rate and n_cust >= 2:
                i1, i2 = random.sample(range(n_cust), 2)
                child[i1], child[i2] = child[i2], child[i1]

            new_pop.append(child)

            d = _get_dist(child)
            if d < best_dist:
                best_dist = d
                best_perm = list(child)

        population = new_pop

    return _get_tour(best_perm)
