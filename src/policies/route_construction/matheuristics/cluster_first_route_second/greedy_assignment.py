"""
Greedy Assignment Module for Fisher & Jaikumar Clustering.

This module implements the original greedy heuristic from Sultana & Akhand (2017)
for assigning nodes to seed clusters in the Cluster-First Route-Second algorithm.

The greedy method sorts all (node, seed) pairs by insertion cost and assigns
nodes to the first feasible seed that respects capacity constraints.

Reference:
    Sultana, T., Akhand, M. A. H., & Rahman, M. M. H. (2017). "A Variant Fisher
    and Jaikumar Algorithm to Solve Capacitated Vehicle Routing Problem".
"""

from typing import Dict, List, Set

import numpy as np


def assign_greedy(
    seeds: List[int],
    mandatory: List[int],
    wastes: Dict[int, float],
    capacity: float,
    distance_matrix: np.ndarray,
    strict_fleet: bool = False,
) -> List[List[int]]:
    """
    Assign nodes to seeds using the greedy heuristic from Sultana & Akhand (2017).

    This implements the original Fisher & Jaikumar greedy assignment strategy:
    1. Calculate insertion cost C_ik = d(i, 0) + d(k, i) - d(k, 0) for all (node, seed) pairs.
    2. Sort all pairs in increasing order of insertion cost.
    3. Assign each node to the first feasible seed (respecting capacity constraints).
    4. Handle unassigned nodes (behavior depends on strict_fleet parameter).
    """
    # Initialize clusters and track capacity usage
    clusters: List[List[int]] = [[] for _ in range(len(seeds))]
    loads = [0.0] * len(seeds)
    assigned: Set[int] = set()

    # Step 1: Pre-assign seeds to their own clusters
    for k_idx, k in enumerate(seeds):
        if k in mandatory:
            clusters[k_idx].append(k)
            loads[k_idx] += wastes.get(k, 0.0)
            assigned.add(k)

    # Step 2: Calculate insertion costs for all (node, seed) pairs
    insertion_costs = []
    for i in mandatory:
        if i in assigned:
            continue
        for k_idx, k in enumerate(seeds):
            # Fisher & Jaikumar insertion cost: C_ik = d(0, i) + d(k, i) - d(k, 0)
            cost = distance_matrix[0, i] + distance_matrix[k, i] - distance_matrix[k, 0]
            insertion_costs.append((cost, i, k_idx))

    # Step 3: Sort by increasing insertion cost (greedy selection)
    insertion_costs.sort(key=lambda x: (x[0], x[1]))

    # Step 4: Assign nodes greedily
    for _cost, node_i, seed_k_idx in insertion_costs:
        if node_i not in assigned:
            waste_i = wastes.get(node_i, 0.0)
            if loads[seed_k_idx] + waste_i <= capacity:
                clusters[seed_k_idx].append(node_i)
                loads[seed_k_idx] += waste_i
                assigned.add(node_i)

    # Step 5: Handle unassigned nodes
    unassigned = [node for node in mandatory if node not in assigned]
    if unassigned:
        _handle_unassigned_nodes(
            unassigned, mandatory, seeds, clusters, loads, assigned, capacity, wastes, strict_fleet
        )

    return clusters


def _handle_unassigned_nodes(
    unassigned: List[int],
    mandatory: List[int],
    seeds: List[int],
    clusters: List[List[int]],
    loads: List[float],
    assigned: Set[int],
    capacity: float,
    wastes: Dict[int, float],
    strict_fleet: bool,
) -> None:
    """Helper to handle nodes that weren't assigned in the initial greedy pass."""
    for node in unassigned:
        waste = wastes.get(node, 0.0)
        fitted = False

        # Try a simple First-Fit into existing clusters
        for k_idx in range(len(seeds)):
            if loads[k_idx] + waste <= capacity:
                clusters[k_idx].append(node)
                loads[k_idx] += waste
                assigned.add(node)
                fitted = True
                break

        if not fitted:
            if strict_fleet:
                # FALLBACK: Try a swap-based local search
                if not _greedy_swap_fallback(node, waste, seeds, clusters, loads, assigned, capacity, wastes):
                    unassigned_final = [n for n in mandatory if n not in assigned]
                    raise ValueError(
                        f"Greedy assignment failed in strict fleet mode. "
                        f"Cannot assign {len(unassigned_final)} nodes to {len(seeds)} vehicles. "
                        f"Unassigned nodes: {unassigned_final}. "
                        f"Consider using exact MIP assignment or increasing fleet capacity."
                    )
            else:
                # FLEXIBLE MODE: Open new vehicle
                seeds.append(node)
                clusters.append([node])
                loads.append(waste)
                assigned.add(node)


def _greedy_swap_fallback(
    node: int,
    waste: float,
    seeds: List[int],
    clusters: List[List[int]],
    loads: List[float],
    assigned: Set[int],
    capacity: float,
    wastes: Dict[int, float],
) -> bool:
    """Attempt to swap an unassigned node with an assigned one to free up capacity."""
    for k_donour in range(len(seeds)):
        for i_idx, node_i in enumerate(clusters[k_donour]):
            if node_i in seeds:
                continue

            waste_i = wastes.get(node_i, 0.0)
            if loads[k_donour] - waste_i + waste <= capacity:
                temp_loads_donour = loads[k_donour] - waste_i + waste
                for k_receiver in range(len(seeds)):
                    if k_receiver != k_donour and loads[k_receiver] + waste_i <= capacity:
                        clusters[k_donour][i_idx] = node
                        clusters[k_receiver].append(node_i)
                        loads[k_donour] = temp_loads_donour
                        loads[k_receiver] += waste_i
                        assigned.add(node)
                        return True
    return False
