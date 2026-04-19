"""
Sector Removal Operator Module.

Calculates the angle of each customer relative to the depot, picks a random
starting angle, and sweeps a sector removing all nodes within that angular
region until ``k`` nodes have been removed.

Also includes profit-based sector removal for VRPP problems.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.destroy.sector import sector_removal
    >>> routes, removed = sector_removal(routes, n_remove=5, coords=coords, depot=(0.5, 0.5))
    >>> from logic.src.policies.helpers.operators.destroy.sector import sector_profit_removal
    >>> routes, removed = sector_profit_removal(routes, n_remove=5, coords=coords, dist_matrix=d,
    ...                                          wastes=w, R=1.0, C=1.0, depot=(0.5, 0.5))
"""

import math
from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np


def sector_removal(
    routes: List[List[int]],
    n_remove: int,
    coords: np.ndarray,
    depot: Tuple[float, float] = (0.0, 0.0),
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes within an angular sector from the depot.

    Converts customer coordinates to polar angles centred on the depot,
    picks a random starting angle, and sweeps until ``n_remove`` nodes
    are collected.

    Args:
        routes: Current solution (list of routes).
        n_remove: Number of nodes to remove.
        coords: Node coordinates, shape ``(N+1, 2)``.  Index 0 = depot.
        depot: Depot coordinates ``(x, y)`` (default uses ``coords[0]``).
        rng: Random number generator.

    Returns:
        Tuple of (modified routes, list of removed node IDs).
    """
    if rng is None:
        rng = Random()

    # Gather nodes and compute angles
    node_angles: List[Tuple[float, int, int, int]] = []  # (angle, node, r_idx, pos)
    depot_x, depot_y = depot

    for r_idx, route in enumerate(routes):
        for pos, node in enumerate(route):
            if node < len(coords):
                dx = float(coords[node, 0]) - depot_x
                dy = float(coords[node, 1]) - depot_y
                angle = math.atan2(dy, dx)
            else:
                angle = 0.0
            node_angles.append((angle, node, r_idx, pos))

    if not node_angles:
        return routes, []

    node_angles.sort(key=lambda x: x[0])
    n_total = len(node_angles)
    n_remove = min(n_remove, n_total)

    start_angle = rng.uniform(-math.pi, math.pi)
    start_idx = 0
    for i, (angle, _, _, _) in enumerate(node_angles):
        if angle >= start_angle:
            start_idx = i
            break

    to_remove: List[Tuple[int, int, int]] = []
    for offset in range(n_total):
        if len(to_remove) >= n_remove:
            break
        idx = (start_idx + offset) % n_total
        _, node, r_idx, pos = node_angles[idx]
        to_remove.append((node, r_idx, pos))

    to_remove.sort(key=lambda x: (x[1], x[2]), reverse=True)
    removed = []
    for node, r_idx, pos in to_remove:
        if pos < len(routes[r_idx]) and routes[r_idx][pos] == node:
            routes[r_idx].pop(pos)
            removed.append(node)

    routes = [r for r in routes if r]
    return routes, removed


def sector_profit_removal(
    routes: List[List[int]],
    n_remove: int,
    coords: np.ndarray,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 1.0,
    C: float = 1.0,
    depot: Tuple[float, float] = (0.0, 0.0),
    bias_low_profit: bool = True,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes within an angular sector, biased toward low-profit regions (VRPP).

    Converts customer coordinates to polar angles centred on the depot, then
    selects a starting angle biased toward sectors with low average marginal profit.
    Sweeps from that angle until ``n_remove`` nodes are collected.

    Args:
        routes: Current solution (list of routes).
        n_remove: Number of nodes to remove.
        coords: Node coordinates, shape ``(N+1, 2)``. Index 0 = depot.
        dist_matrix: Distance matrix.
        wastes: Waste/profit values for each node.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        depot: Depot coordinates ``(x, y)`` (default uses ``coords[0]``).
        bias_low_profit: If True, bias starting angle toward low-profit sectors.
                        If False, use random starting angle (standard sector removal).
        rng: Random number generator.

    Returns:
        Tuple of (modified routes, list of removed node IDs).
    """
    if rng is None:
        rng = Random()

    # 1. Pre-calculate marginal profits
    node_angles_data: List[Tuple[float, int, int, int, float]] = []
    depot_x, depot_y = depot

    for r_idx, route in enumerate(routes):
        for pos, node in enumerate(route):
            # Marginal profit
            revenue = wastes.get(node, 0.0) * R
            prev = 0 if pos == 0 else route[pos - 1]
            nex = 0 if pos == len(route) - 1 else route[pos + 1]
            detour_cost = float(dist_matrix[prev, node] + dist_matrix[node, nex] - dist_matrix[prev, nex])
            profit = revenue - (detour_cost * C)

            # Angle
            if node < len(coords):
                dx = float(coords[node, 0]) - depot_x
                dy = float(coords[node, 1]) - depot_y
                angle = math.atan2(dy, dx)
            else:
                angle = 0.0

            node_angles_data.append((angle, node, r_idx, pos, profit))

    if not node_angles_data:
        return routes, []

    node_angles_data.sort(key=lambda x: x[0])
    n_total = len(node_angles_data)
    n_remove = min(n_remove, n_total)

    # 2. Choose starting angle
    if not bias_low_profit or n_total < 4:
        start_angle = rng.uniform(-math.pi, math.pi)
    else:
        # Divide into quadrants or octants to find low-profit area
        n_sectors = 8
        sector_size = n_total / n_sectors
        sector_scores = []
        for i in range(n_sectors):
            s_idx = int(i * sector_size)
            e_idx = int((i + 1) * sector_size)
            sector_nodes = node_angles_data[s_idx:e_idx]
            if not sector_nodes:
                continue
            avg_p = sum(x[4] for x in sector_nodes) / len(sector_nodes)
            sector_scores.append((avg_p, sector_nodes[0][0]))

        # Pick quadrant/octant with lowest average profit
        sector_scores.sort(key=lambda x: x[0])
        best_sectors = sector_scores[: max(1, n_sectors // 4)]
        start_angle = rng.choice(best_sectors)[1]

    # 3. Sweep
    start_idx = 0
    for i, (angle, _, _, _, _) in enumerate(node_angles_data):
        if angle >= start_angle:
            start_idx = i
            break

    to_remove: List[Tuple[int, int, int]] = []
    for offset in range(n_total):
        if len(to_remove) >= n_remove:
            break
        idx = (start_idx + offset) % n_total
        _, node, r_idx, pos, _ = node_angles_data[idx]
        to_remove.append((node, r_idx, pos))

    to_remove.sort(key=lambda x: (x[1], x[2]), reverse=True)
    removed = []
    for node, r_idx, pos in to_remove:
        if pos < len(routes[r_idx]) and routes[r_idx][pos] == node:
            routes[r_idx].pop(pos)
            removed.append(node)

    routes = [r for r in routes if r]
    return routes, removed
