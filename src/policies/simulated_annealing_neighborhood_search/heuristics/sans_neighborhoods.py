"""
Local search neighborhood operators for SANS-style optimization.

Provides standard neighborhood movement procedures adapted for the SANS policy,
including 2-opt, Or-opt, relocate, and cross-exchange operators. These functions
return modified copies of the input routes to support probabilistic acceptance logic.
"""

import copy
import random

__all__ = [
    "get_neighbors",
    "get_2opt_neighbors",
    "relocate_within_route",
    "cross_exchange",
    "or_opt_move",
    "move_between_routes",
    "insert_bin_in_route",
    "mutate_route_by_swapping_bins",
]


def get_neighbors(route: list) -> list:
    """
    Generate the neighborhood of a route through node swaps.

    Args:
        route: Current route.

    Returns:
        List of neighboring routes.
    """
    neighbors = []
    if len(route) <= 3:
        return [route[:]]

    for i in range(1, len(route) - 2):
        for j in range(i + 1, len(route) - 1):
            new_route = route[:]
            new_route[i], new_route[j] = new_route[j], new_route[i]
            neighbors.append(new_route)
    return neighbors


def get_2opt_neighbors(route: list) -> list:
    """
    Generate 2-opt neighbors for a route.

    Args:
        route: Proposed node sequence.

    Returns:
        Candidate neighboring routes.
    """
    neighbors = []
    n = len(route)
    if n <= 4:
        return [route[:]]

    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            new_route = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
            neighbors.append(new_route)
    return neighbors


def relocate_within_route(route: list) -> list:
    """
    Relocate a random bin to a different position in the same route.

    Args:
        route: Node sequence.

    Returns:
        Mutated route.
    """
    if len(route) <= 3:
        return route[:]
    new_route = route[:]
    i, j = sorted(random.sample(range(1, len(route) - 1), 2))
    bin_moved = new_route.pop(i)
    new_route.insert(j, bin_moved)
    return new_route


def cross_exchange(routes: list) -> list:
    """
    Exchange segments between two different routes.

    Args:
        routes: Set of routes.

    Returns:
        Mutated solution.
    """
    if len(routes) < 2:
        return [r[:] for r in routes]

    new_routes = copy.deepcopy(routes)
    r1, r2 = random.sample(range(len(new_routes)), 2)
    route1, route2 = new_routes[r1], new_routes[r2]
    if len(route1) <= 3 or len(route2) <= 3:
        return new_routes

    i1 = random.randint(1, len(route1) - 2)
    i2 = random.randint(1, len(route2) - 2)

    new_route1 = route1[:i1] + route2[i2:-1] + [0]
    new_route2 = route2[:i2] + route1[i1:-1] + [0]

    new_routes[r1] = new_route1
    new_routes[r2] = new_route2
    return new_routes


def or_opt_move(route: list) -> list:
    """
    Apply Or-opt operator (move a segment of 1-2 nodes) within a route.

    Args:
        route: Node sequence.

    Returns:
        Mutated route.
    """
    if len(route) <= 4:
        return route[:]

    new_route = route[:]
    k = random.choice([1, 2])
    start = random.randint(1, len(route) - k - 1)
    segment = new_route[start : start + k]
    del new_route[start : start + k]

    insert_pos = random.randint(1, len(new_route) - 1)
    new_route = new_route[:insert_pos] + segment + new_route[insert_pos:]
    return new_route


def move_between_routes(routes: list, data, vehicle_capacity: float, id_to_index: dict) -> list:
    """
    Move a random bin from one route to another, respecting capacity constraints.

    Args:
        routes: Set of routes.
        data: Bin weights data.
        vehicle_capacity: Tanker capacity.
        id_to_index: Mapping.

    Returns:
        List of mutated solution candidates.
    """
    moves = []
    # Import locally to avoid circularity if any
    try:
        stocks = dict(zip(data["#bin"], data["Stock"]))
    except (KeyError, TypeError):
        # Fallback if data format is different
        return []

    for i in range(len(routes)):
        for j in range(len(routes)):
            if i == j:
                continue
            for idx in range(1, len(routes[i]) - 1):
                bin_to_move = routes[i][idx]
                if bin_to_move == 0:
                    continue
                load_j = sum(stocks.get(b, 0) for b in routes[j] if b != 0)
                if load_j + stocks.get(bin_to_move, 0) > vehicle_capacity:
                    continue
                new_routes = copy.deepcopy(routes)
                new_routes[i].pop(idx)
                insert_pos = random.randint(1, len(new_routes[j]) - 1)
                new_routes[j].insert(insert_pos, bin_to_move)
                moves.append(new_routes)
    return moves


def insert_bin_in_route(route: list, bin_id: int, id_to_index: dict, distance_matrix) -> list:
    """
    Insert a bin into a route at the position minimizing cost increase.

    Args:
        route: Current route.
        bin_id: Bin to insert.
        id_to_index: Mapping from bin ID to matrix index.
        distance_matrix: Distance matrix.

    Returns:
        New route with the inserted bin.
    """
    best_route = route[:]
    best_increase = float("inf")

    # Start and end positions for insertion (between depots)
    for i in range(1, len(route)):
        prev_node = route[i - 1]
        next_node = route[i]

        idx_prev = id_to_index.get(prev_node, prev_node)
        idx_next = id_to_index.get(next_node, next_node)
        idx_bin = id_to_index.get(bin_id, bin_id)

        try:
            increase = (
                distance_matrix[idx_prev][idx_bin]
                + distance_matrix[idx_bin][idx_next]
                - distance_matrix[idx_prev][idx_next]
            )
        except (IndexError, TypeError):
            continue

        if increase < best_increase:
            best_increase = increase
            best_route = route[:i] + [bin_id] + route[i:]

    return best_route


def mutate_route_by_swapping_bins(route: list, num_bins: int = 1) -> list:
    """
    Swap random nodes within a route.

    Args:
        route: Route to mutate.
        num_bins: Number of swap operations to perform.

    Returns:
        Mutated route.
    """
    if len(route) <= 3:
        return route[:]

    new_route = route[:]
    for _ in range(num_bins):
        if len(new_route) > 3:
            i, j = random.sample(range(1, len(new_route) - 1), 2)
            new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route
