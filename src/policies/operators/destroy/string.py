"""
String Removal Operator Module.

This module implements the string removal heuristic, which removes contiguous
sequences (strings) of nodes from routes to preserve local structure while
creating gaps for re-insertion.

Attributes:
    None

Example:
    >>> from logic.src.policies.operators.destroy.string import string_removal
    >>> routes, removed = string_removal(routes, n_remove=5, ...)
"""

import random
from typing import List, Tuple

import numpy as np


def string_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    max_string_len: int = 4,
    avg_string_len: float = 3.0,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove contiguous strings of customers to induce spatial slack.

    The key insight of SISR is that removing adjacent customers creates a large
    contiguous "hole" in the route, providing maneuverability for reinsertion.
    This also reduces the number of re-insertions needed compared to random removal.

    Args:
        routes: Current routes.
        n_remove: Number of nodes to remove.
        dist_matrix: Distance matrix (used for propagation).
        max_string_len: Maximum length of a string to remove.
        avg_string_len: Average string length (unused, kept for API compatibility).

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    if not any(routes) or n_remove <= 0:
        return routes, []

    removed: List[int] = []
    max_iter = n_remove * 3  # Prevent infinite loops
    iterations = 0

    while len(removed) < n_remove and iterations < max_iter:
        iterations += 1

        # Pick a random seed from remaining nodes
        available_routes = [(i, r) for i, r in enumerate(routes) if r]
        if not available_routes:
            break

        r_idx, route = random.choice(available_routes)
        if not route:
            continue

        # Pick seed position
        seed_pos = random.randint(0, len(route) - 1)
        seed_node = route[seed_pos]

        if seed_node in removed:
            continue

        # Determine string length (geometric-like distribution)
        # L ~ 1 + geometric(1/avg_string_len)
        string_len = 1
        while string_len < max_string_len and random.random() < (1 - 1 / avg_string_len):
            string_len += 1

        # Don't remove more than needed
        remaining = n_remove - len(removed)
        string_len = min(string_len, remaining, len(route))

        # Extract string starting at seed_pos
        start = seed_pos
        end = min(seed_pos + string_len, len(route))
        string_nodes = route[start:end]

        # Remove string from route (reverse order to maintain indices)
        for pos in range(end - 1, start - 1, -1):
            node = routes[r_idx].pop(pos)
            removed.append(node)

        # Propagate to neighbors: remove strings from adjacent routes
        if len(removed) < n_remove and string_nodes:
            _propagate_string_removal(routes, removed, dist_matrix, string_nodes, n_remove, max_string_len)

    # Clean empty routes
    routes = [r for r in routes if r]
    return routes, removed


def _propagate_string_removal(
    routes: List[List[int]],
    removed: List[int],
    dist_matrix: np.ndarray,
    seed_nodes: List[int],
    n_remove: int,
    max_string_len: int,
) -> None:
    """
    Propagate string removal to neighboring routes.

    After removing a string, look at the spatial neighbors of the removed nodes
    and remove strings from their routes as well. This creates a concentrated
    "disaster zone" across multiple routes.
    """
    # Find neighbors of removed string
    neighbor_candidates = []
    for seed in seed_nodes:
        if seed >= len(dist_matrix):
            continue
        distances = dist_matrix[seed]
        for node_id in range(1, len(distances)):
            if node_id not in removed and node_id not in seed_nodes:
                neighbor_candidates.append((node_id, distances[node_id]))

    # Sort by distance
    neighbor_candidates.sort(key=lambda x: x[1])

    # Take closest neighbors
    for neighbor, _ in neighbor_candidates[:3]:
        if len(removed) >= n_remove:
            break

        # Find which route contains this neighbor
        for _r_idx, route in enumerate(routes):
            if neighbor in route:
                pos = route.index(neighbor)
                # Remove a small string around this neighbor
                string_len = min(2, len(route), n_remove - len(removed))
                start = max(0, pos - string_len // 2)
                end = min(len(route), start + string_len)

                for p in range(end - 1, start - 1, -1):
                    if route[p] not in removed:
                        removed.append(route.pop(p))
                break
