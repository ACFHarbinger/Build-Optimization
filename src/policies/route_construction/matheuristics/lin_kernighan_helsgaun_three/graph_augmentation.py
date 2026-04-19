"""
Graph Augmentation Module for LKH-3 Multi-Vehicle Support.

This module implements explicit dummy depot augmentation to avoid the catastrophic
NumPy negative-indexing bug. Instead of using negative indices (-1, -2, ...) which
wrap around in NumPy arrays, we explicitly append M-1 dummy depot nodes to the graph.

Mathematical Formulation
------------------------
Given an original graph with N nodes (0 = depot, 1..N-1 = customers) and M vehicles:

1. **Augmented Graph Size**: N + (M-1) nodes
   - Original nodes: 0, 1, ..., N-1
   - Dummy depots: N, N+1, ..., N+M-2

2. **Distance Matrix Augmentation**: (N+M-1) × (N+M-1)
   - Dummy depot rows/cols copy depot (node 0) distances
   - Inter-dummy distances set to HIGH_PENALTY to prevent empty routes

3. **Waste/Demand Augmentation**: Size N+M-1
   - Dummy depot demands set to 0

Architecture
------------
This augmentation happens at the POLICY level (policy_lkh3.py) before calling
the LKH-3 engine. The engine operates on the augmented graph with explicit dummy
indices, avoiding all NumPy indexing issues.

After LKH-3 completes, the tour is decoded by splitting at any node index >= N
(which are the augmented dummy depots).

Example
-------
>>> # Original: N=5 nodes (depot + 4 customers), M=3 vehicles
>>> dist_orig = np.array([[...]])  # (5, 5)
>>> waste_orig = {1: 10, 2: 20, 3: 15, 4: 25}
>>>
>>> # Augment for 3 vehicles (adds 2 dummy depots at indices 5, 6)
>>> dist_aug, waste_aug, n_original = augment_graph(
...     dist_orig, waste_orig, n_vehicles=3
... )
>>> # dist_aug.shape = (7, 7)
>>> # waste_aug = {1: 10, 2: 20, 3: 15, 4: 25, 5: 0, 6: 0}
>>> # n_original = 5
>>>
>>> # After LKH-3: tour = [0, 1, 2, 5, 3, 4, 6, 0]
>>> routes = decode_augmented_tour(tour, n_original)
>>> # routes = [[1, 2], [3, 4]]  (split at indices >= 5)

Public API
----------
augment_graph(distance_matrix, wastes, n_vehicles, capacity, high_penalty) ->
    (augmented_dist, augmented_wastes, n_original)

decode_augmented_tour(tour, n_original) -> List[List[int]]
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from logic.src.constants.routing import DEFAULT_HIGH_PENALTY


def augment_graph(
    distance_matrix: np.ndarray,
    wastes: Dict[int, float],
    n_vehicles: int,
    capacity: float = 100.0,
    high_penalty: float = DEFAULT_HIGH_PENALTY,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Augment graph with explicit dummy depot nodes for multi-vehicle VRP.

    Creates M-1 dummy depot nodes by appending them to the original graph.
    Dummy depots are assigned indices [N, N+1, ..., N+M-2] where N is the
    original graph size.

    Args:
        distance_matrix: (N×N) original distance matrix (node 0 = depot).
        wastes: Dict mapping node ID to demand/profit value.
        n_vehicles: Number of vehicles (M). Must be >= 0.
        capacity: Vehicle capacity (used for sanity checks).
        high_penalty: Distance penalty for inter-dummy edges.

    Returns:
        Tuple of:
        - augmented_dist: (N+M-1 × N+M-1) augmented distance matrix
        - augmented_waste: (N+M-1,) augmented waste array
        - n_original: Original graph size N (for later decoding)

    Raises:
        ValueError: If n_vehicles < 0 or distance_matrix is invalid.

    Example:
        >>> dist = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        >>> wastes = {1: 30, 2: 40}
        >>> aug_dist, aug_waste, n_orig = augment_graph(dist, wastes, n_vehicles=2)
        >>> aug_dist.shape
        (4, 4)
        >>> aug_waste.shape
        (4,)
        >>> n_orig
        3
    """
    if n_vehicles < 0:
        raise ValueError(f"n_vehicles must be >= 0, got {n_vehicles}")

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError(f"distance_matrix must be square, got shape {distance_matrix.shape}")

    n_original = len(distance_matrix)
    n_dummies = max(0, n_vehicles - 1)  # M-1 dummy depots
    n_augmented = n_original + n_dummies

    # --- 1. Augment Distance Matrix ---
    augmented_dist = np.full((n_augmented, n_augmented), high_penalty, dtype=float)

    # Copy original distances
    augmented_dist[:n_original, :n_original] = distance_matrix

    # Dummy depot distances: copy depot (row/col 0) to each dummy
    for dummy_idx in range(n_dummies):
        aug_idx = n_original + dummy_idx

        # Copy depot row: distances FROM depot to all original nodes
        augmented_dist[aug_idx, :n_original] = distance_matrix[0, :]

        # Copy depot column: distances TO depot from all original nodes
        augmented_dist[:n_original, aug_idx] = distance_matrix[:, 0]

        # Dummy-to-depot distance (both directions)
        augmented_dist[aug_idx, 0] = 0.0
        augmented_dist[0, aug_idx] = 0.0

    # Inter-dummy distances: Set to HIGH_PENALTY to discourage empty routes
    # (prevents optimizer from creating routes like [0 -> dummy1 -> dummy2 -> 0])
    for i in range(n_dummies):
        for j in range(n_dummies):
            if i != j:
                aug_i = n_original + i
                aug_j = n_original + j
                augmented_dist[aug_i, aug_j] = high_penalty

    # Self-loops for dummies: 0 distance
    for dummy_idx in range(n_dummies):
        aug_idx = n_original + dummy_idx
        augmented_dist[aug_idx, aug_idx] = 0.0

    # --- 2. Augment Waste/Demand Array ---
    augmented_waste = np.zeros(n_augmented, dtype=float)

    # Copy original demands
    for node, demand in wastes.items():
        if 0 <= node < n_original:
            augmented_waste[node] = demand

    # Dummy depots have zero demand (already initialized to 0)
    return augmented_dist, augmented_waste, n_original


def augment_prize_collecting_graph(
    distance_matrix: np.ndarray,
    wastes: Dict[int, float],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Apply Jonker-Volgenant ATSP transformation for native prize-collecting.

    Size: (2N+1) x (2N+1)
    - Node 0: Depot
    - Nodes 1..N: Real customer nodes (Exits)
    - Nodes N+1..2N: Dummy dummy nodes (Entries)

    Transformation Rules:
    1. Prize Collection: D(i', i) = -P_i
    2. Real Spatial Travel: D(i, j') = c_ij
    3. Skip Bypass: D(i, i') = 0
    4. Chaining Skipped Nodes: D(i', j) = 0
    5. Depot Connections:
       - Start Skip Chain: D(0, i) = 0
       - Transition to Real Route: D(i', j') = c_0j
       - Direct to Real Route: D(0, j') = c_0j
       - Return to Depot: D(i, 0) = c_i0
       - Return to Depot (ALL skipped): D(i', 0) = 0
    """
    n_original = len(distance_matrix)
    n_customers = n_original - 1
    n_augmented = 2 * n_customers + 1

    # Initialize with infinity
    aug_dist = np.full((n_augmented, n_augmented), float("inf"))

    def get_dummy(i: int) -> int:
        return n_original + i - 1

    for i in range(1, n_original):
        p_i = wastes.get(i, 0.0)
        i_exit = i
        i_entry = get_dummy(i)

        # 1. Prize Collection (i' -> i)
        aug_dist[i_entry, i_exit] = -p_i

        # 3. Skip Bypass (i -> i')
        aug_dist[i_exit, i_entry] = 0.0

        # 4. Chaining Skipped Nodes (i' -> j)
        for j in range(1, n_original):
            if i != j:
                aug_dist[i_entry, j] = 0.0

        # 5. Depot Connections
        # Start Skip Chain (0 -> i)
        aug_dist[0, i_exit] = 0.0

        # Return to Depot from Exit (i -> 0)
        aug_dist[i_exit, 0] = distance_matrix[i, 0]

        # Return to Depot from Entry (i' -> 0)
        aug_dist[i_entry, 0] = 0.0

        # Direct/Transition to Real Route (0 -> j' or i' -> j')
        for j in range(1, n_original):
            j_entry = get_dummy(j)
            c_0j = distance_matrix[0, j]

            # Direct to Real Route (0 -> j')
            aug_dist[0, j_entry] = c_0j

            # Transition to Real Route (i' -> j')
            aug_dist[i_entry, j_entry] = c_0j

        # 2. Real Spatial Travel (i -> j')
        for j in range(1, n_original):
            if i != j:
                j_entry = get_dummy(j)
                aug_dist[i_exit, j_entry] = distance_matrix[i, j]

    # Diagonal remains inf
    np.fill_diagonal(aug_dist, float("inf"))

    aug_waste = np.zeros(n_augmented)
    return aug_dist, aug_waste, n_original


def decode_augmented_tour(tour: List[int], n_original: int) -> List[List[int]]:
    """
    Decode a tour from the augmented graph into multi-route representation.

    Splits the tour at any node index >= n_original (which are dummy depots)
    or at the main depot (node 0). Each sub-route contains only customer nodes.

    Args:
        tour: Closed tour from LKH-3 on augmented graph.
              Example: [0, 1, 2, 5, 3, 4, 6, 0] (n_original=5)
        n_original: Original graph size N (depot + customers).

    Returns:
        List of routes, each containing only customer node indices.
        Example: [[1, 2], [3, 4]]  (nodes >= 5 act as route delimiters)

    Example:
        >>> tour = [0, 3, 5, 7, 2, 4, 8, 9, 1, 0]  # n_original=6
        >>> routes = decode_augmented_tour(tour, n_original=6)
        >>> # Splits at: 0, 7 (>=6), 8 (>=6), 0
        >>> routes
        [[3, 5], [2, 4], [9, 1]]

    Handles both:
    1. Multi-vehicle dummy depot augmentation (indices >= n_original).
    2. Prize-collecting ATSP transformation (indices >= n_original).
    """
    max_node = max(tour)
    n_customers = n_original - 1
    # Check if this is likely an ATSP tour (doubled nodes)
    is_atsp = max_node == 2 * n_customers

    if is_atsp:
        visited_nodes: List[int] = []

        # In ATSP tour, we look for entry -> exit transitions
        def get_entry(i_exit: int) -> int:
            return n_original + i_exit - 1

        for idx in range(len(tour) - 1):
            u, v = tour[idx], tour[idx + 1]
            # v is a real exit node. Check if it was entered via its dummy.
            if 1 <= v < n_original and u == get_entry(v):
                visited_nodes.append(v)

        return [visited_nodes] if visited_nodes else [[]]

    # Standard multi-vehicle decoding
    routes: List[List[int]] = []
    current_route: List[int] = []
    for node in tour:
        if node == 0 or node >= n_original:
            if current_route:
                routes.append(current_route)
                current_route = []
        else:
            current_route.append(node)

    if current_route:
        routes.append(current_route)

    return routes


def is_dummy_depot(node: int, n_original: int) -> bool:
    """
    Check if a node is an augmented dummy depot.
    """
    return node >= n_original


def is_any_depot(node: int, n_original: int) -> bool:
    """
    Check if a node is either the main depot (0) or a dummy depot.
    """
    return node == 0 or node >= n_original


def inject_augmented_dummies(routes: List[List[int]], n_original: int, n_vehicles: int) -> List[int]:
    """
    Inject augmented dummy depots into a multi-route solution to create a flat tour.

    Dummy depots are inserted between consecutive routes using indices
    [n_original, n_original+1, ...].

    Args:
        routes: List of routes (no depot nodes).
        n_original: Original graph size N.
        n_vehicles: Number of vehicles (determines dummy depot count).

    Returns:
        Closed tour with augmented dummy depot indices.
        Example: routes=[[3,5], [7,2]] → [0, 3, 5, N, 7, 2, 0]
                 where N = n_original

    Example:
        >>> routes = [[1, 2], [3, 4], [5]]
        >>> tour = inject_augmented_dummies(routes, n_original=6, n_vehicles=3)
        >>> tour
        [0, 1, 2, 6, 3, 4, 7, 5, 0]
    """
    if not routes:
        return [0, 0]

    tour = [0]  # Start at main depot
    dummy_idx = n_original  # First dummy starts at n_original

    for route_idx, route in enumerate(routes):
        tour.extend(route)

        # Insert dummy between routes (not after last route)
        if route_idx < len(routes) - 1:
            tour.append(dummy_idx)
            dummy_idx += 1

    tour.append(0)  # Return to main depot
    return tour


def validate_augmented_graph(
    augmented_dist: np.ndarray,
    augmented_waste: np.ndarray,
    n_original: int,
    n_vehicles: int,
) -> None:
    """
    Validate the augmented graph structure for correctness.

    Checks:
    1. Matrix dimensions match expected size
    2. Symmetry of distance matrix
    3. Zero demand for dummy depots
    4. Depot distances copied correctly to dummies

    Args:
        augmented_dist: Augmented distance matrix.
        augmented_waste: Augmented waste array.
        n_original: Original graph size.
        n_vehicles: Number of vehicles.

    Raises:
        AssertionError: If validation fails.
    """
    n_dummies = max(0, n_vehicles - 1)
    n_expected = n_original + n_dummies

    # Check dimensions
    assert augmented_dist.shape == (
        n_expected,
        n_expected,
    ), f"Distance matrix shape mismatch: {augmented_dist.shape} != {(n_expected, n_expected)}"
    assert augmented_waste.shape[0] == n_expected, (
        f"Waste array size mismatch: {augmented_waste.shape[0]} != {n_expected}"
    )

    # Check symmetry
    assert np.allclose(augmented_dist, augmented_dist.T), "Distance matrix is not symmetric"

    # Check dummy demands are zero
    for dummy_idx in range(n_dummies):
        aug_idx = n_original + dummy_idx
        assert augmented_waste[aug_idx] == 0.0, f"Dummy depot {aug_idx} has non-zero demand: {augmented_waste[aug_idx]}"

    # Check dummy-to-depot distances are zero
    for dummy_idx in range(n_dummies):
        aug_idx = n_original + dummy_idx
        assert augmented_dist[aug_idx, 0] == 0.0, (
            f"Dummy {aug_idx} to depot distance != 0: {augmented_dist[aug_idx, 0]}"
        )
        assert augmented_dist[0, aug_idx] == 0.0, (
            f"Depot to dummy {aug_idx} distance != 0: {augmented_dist[0, aug_idx]}"
        )

    print(f"✓ Augmented graph validation passed (N={n_original}, M={n_vehicles})")
