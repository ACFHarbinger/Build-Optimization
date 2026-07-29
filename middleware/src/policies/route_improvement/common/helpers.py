"""
Shared helpers for route improvers.

Kept deliberately small and stateless so route improvers can copy import
this module without coupling themselves to each other.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


def resolve_mandatory_nodes(
    kwargs: Dict[str, Any],
    config: Dict[str, Any],
) -> Optional[List[int]]:
    """
    Resolve the list of mandatory-node IDs from the route improver context.

    Checks keys in priority order:
        1. kwargs["mandatory_nodes"] — explicit caller override
        2. kwargs["mandatory"]         — mandatory selection action output
        3. config["mandatory_nodes"] — config file default

    Returns None when no source is set (the canonical operators treat None
    as "no mandatory nodes", so this is a safe default).
    """
    if "mandatory_nodes" in kwargs and kwargs["mandatory_nodes"] is not None:
        return list(kwargs["mandatory_nodes"])
    if "mandatory" in kwargs and kwargs["mandatory"] is not None:
        return list(kwargs["mandatory"])
    if "mandatory_nodes" in config and config["mandatory_nodes"] is not None:
        return list(config["mandatory_nodes"])
    return None


def upgrade_repair_op_to_profit(repair_op: str, revenue_kg: float, cost_per_km: float) -> str:
    """
    Return the profit-aware variant of a repair operator when economic
    parameters are set, otherwise return the operator unchanged.

    Auto-upgrade is idempotent: passing an operator that already has
    "profit" in its name returns the name unchanged.
    """
    if (revenue_kg > 0 or cost_per_km > 0) and "profit" not in repair_op:
        return repair_op + "_profit"
    return repair_op


def to_numpy(distance_matrix: Any) -> np.ndarray:
    """Safely convert distance matrix to numpy array, handling Torch tensors (including CUDA)."""
    if isinstance(distance_matrix, np.ndarray):
        return distance_matrix
    if hasattr(distance_matrix, "cpu"):
        return distance_matrix.cpu().numpy()
    return np.array(distance_matrix)


def split_tour(tour: Sequence[int]) -> List[List[int]]:
    """Split a multi-trip tour separated by depot 0s into per-trip lists."""
    routes: List[List[int]] = []
    current: List[int] = []
    for node in tour:
        if node == 0:
            if current:
                routes.append(current)
                current = []
        else:
            current.append(node)
    if current:
        routes.append(current)
    return routes


def assemble_tour(routes: Sequence[Sequence[int]]) -> List[int]:
    """Reassemble per-trip routes into a depot-separated tour."""
    result: List[int] = [0]
    for r in routes:
        if r:
            result.extend(r)
            result.append(0)
    return result


def route_distance(route: Sequence[int], dm: np.ndarray) -> float:
    """Length of a single route, including depot-out and depot-return legs."""
    if not route:
        return 0.0
    total = float(dm[0, route[0]]) + float(dm[route[-1], 0])
    for i in range(len(route) - 1):
        total += float(dm[route[i], route[i + 1]])
    return total


def tour_distance(routes: Iterable[Sequence[int]], dm: np.ndarray) -> float:
    """Total distance over all routes."""
    return sum(route_distance(r, dm) for r in routes)


def route_load(route: Sequence[int], wastes: Dict[int, float]) -> float:
    """Sum of waste masses for a single route, given a {node_id: mass} dict."""
    return sum(float(wastes.get(n, 0.0)) for n in route)
