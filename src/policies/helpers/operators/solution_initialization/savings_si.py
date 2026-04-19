"""
Clarke-Wright Savings Algorithm for VRPP initialization.

Reference:
    Clarke, G., & Wright, J. W. (1964). Scheduling of vehicles from a central
    depot to a number of delivery points. Operations Research, 12(4), 568-581.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.utils.policy.routes import (
    prune_unprofitable_routes,
)


def _compute_savings(eligible: List[int], dist_matrix: np.ndarray) -> List[Tuple[float, int, int]]:
    """Compute and sort all positive savings pairs."""
    savings = []
    for idx, i in enumerate(eligible):
        for j in eligible[idx + 1 :]:
            s = dist_matrix[0, i] + dist_matrix[0, j] - dist_matrix[i, j]
            if s > 0:
                savings.append((s, i, j))
    savings.sort(reverse=True)
    return savings


def _try_merge(
    s: float,
    i: int,
    j: int,
    route_of: Dict[int, List[int]],
    route_key: Dict[int, int],
    load_of: Dict[int, float],
    capacity: float,
) -> bool:
    """Attempt to merge routes containing i and j. Returns True if merged."""
    ki, kj = route_key[i], route_key[j]
    if ki == kj:
        return False

    ri, rj = route_of[ki], route_of[kj]
    if load_of[ki] + load_of[kj] > capacity:
        return False

    merged = None
    if ri[-1] == i and rj[0] == j:
        merged = ri + rj
    elif ri[-1] == i and rj[-1] == j:
        merged = ri + list(reversed(rj))
    elif ri[0] == i and rj[0] == j:
        merged = list(reversed(ri)) + rj
    elif ri[0] == i and rj[-1] == j:
        merged = list(reversed(ri)) + list(reversed(rj))

    if merged is None:
        return False

    new_key = merged[0]
    new_load = load_of[ki] + load_of[kj]
    for node in merged:
        route_of[node] = merged
        route_key[node] = new_key
        load_of[node] = new_load
    return True


def build_savings_routes(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
) -> List[List[int]]:
    """
    Build routes using the Clarke-Wright Savings Algorithm.
    """
    mandatory_set = set(mandatory_nodes or [])
    nodes = list(range(1, len(dist_matrix)))
    eligible = [
        i for i in nodes if i in mandatory_set or wastes.get(i, 0.0) * R >= (dist_matrix[0, i] + dist_matrix[i, 0]) * C
    ]

    if not eligible:
        return []

    savings = _compute_savings(eligible, dist_matrix)
    route_of = {i: [i] for i in eligible}
    load_of = {i: wastes.get(i, 0.0) for i in eligible}
    route_key = {i: i for i in eligible}

    for s, i, j in savings:
        _try_merge(s, i, j, route_of, route_key, load_of, capacity)

    seen_keys = set()
    routes = []
    for i in eligible:
        k = route_key[i]
        if k not in seen_keys:
            seen_keys.add(k)
            routes.append(route_of[k])

    return prune_unprofitable_routes(routes, dist_matrix, wastes, R, C, mandatory_set)
