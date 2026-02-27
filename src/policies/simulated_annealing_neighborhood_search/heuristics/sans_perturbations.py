"""
SANS-specific mutation and perturbation operators.

Implements specialized operators for the SANS policy variant, including
destroy and repair heuristics (add/remove bins) and multi-node relocations
(n-move, n-swap) in both random and consecutive modes.
"""

import copy
import random

from .sans_neighborhoods import insert_bin_in_route

__all__ = [
    "remove_bins_from_route",
    "move_n_route_random",
    "swap_n_route_random",
    "remove_n_bins_random",
    "add_n_bins_random",
    "add_route_with_removed_bins_random",
    "move_n_route_consecutive",
    "swap_n_route_consecutive",
    "remove_n_bins_consecutive",
    "add_n_bins_consecutive",
    "add_route_with_removed_bins_consecutive",
]


def remove_bins_from_route(route: list, num_bins: int = 1) -> list:
    """
    Remove random bins from a single route.

    Args:
        route: Route to mutate.
        num_bins: Number of bins to remove.

    Returns:
        New route with bins removed.
    """
    if len(route) <= 2:
        return route[:]
    new_route = route[:]
    num_to_remove = min(num_bins, len(new_route) - 2)
    indices = random.sample(range(1, len(new_route) - 1), num_to_remove)
    for idx in sorted(indices, reverse=True):
        new_route.pop(idx)
    return new_route


def move_n_route_random(routes_list: list, n: int = 2) -> list:
    """
    Move n random bins from one route to another.

    Args:
        routes_list: Current routing solution.
        n: Number of nodes to move.

    Returns:
        Mutated routing solution.
    """
    new_routes = copy.deepcopy(routes_list)
    non_empty = [r for r in new_routes if len(r) > 2]
    if not non_empty:
        return new_routes

    donor_route = random.choice(non_empty)
    if len(donor_route) < n + 2:
        return new_routes

    nodes_to_move = random.sample(donor_route[1:-1], n)
    # Filter donor route
    donor_idx = new_routes.index(donor_route)
    new_routes[donor_idx] = [x for x in donor_route if x not in nodes_to_move]

    # Receptor route
    receptor_route = random.choice([r for r in new_routes if len(r) >= 2])
    receptor_idx = new_routes.index(receptor_route)
    insert_pos = random.randint(1, max(1, len(receptor_route) - 1))

    # Insert nodes (in order)
    new_routes[receptor_idx] = receptor_route[:insert_pos] + nodes_to_move + receptor_route[insert_pos:]
    return new_routes


def swap_n_route_random(routes_list: list, n: int = 2) -> list:
    """
    Swap n random bins between two distinct routes.

    Args:
        routes_list: Current routing solution.
        n: Number of nodes to swap.

    Returns:
        Mutated routing solution.
    """
    if len(routes_list) < 2:
        return copy.deepcopy(routes_list)

    new_routes = copy.deepcopy(routes_list)
    r1_idx, r2_idx = random.sample(range(len(new_routes)), 2)
    route1, route2 = new_routes[r1_idx], new_routes[r2_idx]

    if len(route1) < n + 2 or len(route2) < n + 2:
        return new_routes

    nodes1 = random.sample(route1[1:-1], n)
    nodes2 = random.sample(route2[1:-1], n)

    # Perform swap
    new_routes[r1_idx] = [x for x in route1 if x not in nodes1] + nodes2
    new_routes[r2_idx] = [x for x in route2 if x not in nodes2] + nodes1

    # Keep depots (this implementation is a bit rough, let's fix it by maintaining depots)
    new_routes[r1_idx] = [0] + [x for x in new_routes[r1_idx] if x != 0] + [0]
    new_routes[r2_idx] = [0] + [x for x in new_routes[r2_idx] if x != 0] + [0]

    return new_routes


def remove_n_bins_random(routes_list: list, removed_bins: set, bins_cannot_removed: set, n: int = 2) -> list:
    """
    Remove n random bins from routes and add them to the removed pool.

    Args:
        routes_list: Current routes.
        removed_bins: Set of removed bins to update.
        bins_cannot_removed: Bins that must stay in routes.
        n: Number of bins to remove.

    Returns:
        Mutated solution.
    """
    new_routes = copy.deepcopy(routes_list)
    all_removable = []
    for r in new_routes:
        for b in r[1:-1]:
            if b not in bins_cannot_removed:
                all_removable.append(b)

    if not all_removable:
        return new_routes

    to_remove = random.sample(all_removable, min(n, len(all_removable)))
    for b in to_remove:
        removed_bins.add(b)
        for _i, r in enumerate(new_routes):
            if b in r:
                r.remove(b)
                break
    return new_routes


def add_n_bins_random(
    routes_list: list,
    removed_bins: set,
    stocks: dict,
    vehicle_capacity: float,
    id_to_index: dict,
    distance_matrix,
    n: int = 2,
) -> list:
    """
    Add n random bins from the removed pool back into routes.
    """
    new_routes = copy.deepcopy(routes_list)
    if not new_routes or not removed_bins:
        return new_routes

    bins_to_add = random.sample(list(removed_bins), min(n, len(removed_bins)))
    for b in bins_to_add:
        # Try a random route
        r_idx = random.randint(0, len(new_routes) - 1)
        route = new_routes[r_idx]
        load = sum(stocks.get(x, 0) for x in route if x != 0)
        if load + stocks.get(b, 0) <= vehicle_capacity:
            new_routes[r_idx] = insert_bin_in_route(route, b, id_to_index, distance_matrix)
            removed_bins.discard(b)
    return new_routes


def add_route_with_removed_bins_random(
    routes_list: list, removed_bins: set, stocks: dict, vehicle_capacity: float
) -> list:
    """
    Create a new route from random bins in the removed pool.
    """
    if not removed_bins:
        return copy.deepcopy(routes_list)

    new_routes = copy.deepcopy(routes_list)
    new_route = [0]
    current_load = 0

    available = list(removed_bins)
    random.shuffle(available)

    for b in available:
        b_weight = stocks.get(b, 0)
        if current_load + b_weight <= vehicle_capacity:
            new_route.append(b)
            current_load += b_weight
            removed_bins.discard(b)

    if len(new_route) > 1:
        new_route.append(0)
        new_routes.append(new_route)

    return new_routes


def move_n_route_consecutive(routes_list: list, n: int = 2) -> list:
    """
    Move a sequence of consecutive bins from one route to another.
    """
    new_routes = copy.deepcopy(routes_list)
    non_empty = [r for r in new_routes if len(r) >= n + 2]
    if not non_empty:
        return new_routes

    donor_route = random.choice(non_empty)
    donor_idx = new_routes.index(donor_route)
    start_idx = random.randint(1, len(donor_route) - n - 1)
    segment = donor_route[start_idx : start_idx + n]

    # Remove from donor
    new_donor = donor_route[:start_idx] + donor_route[start_idx + n :]
    new_routes[donor_idx] = new_donor

    # Receptor
    receptor_route = random.choice(new_routes)
    receptor_idx = new_routes.index(receptor_route)
    insert_pos = random.randint(1, max(1, len(receptor_route) - 1))
    new_routes[receptor_idx] = receptor_route[:insert_pos] + segment + receptor_route[insert_pos:]

    return new_routes


def swap_n_route_consecutive(routes_list: list, n: int = 2) -> list:
    """
    Swap consecutive segments between two routes.
    """
    if len(routes_list) < 2:
        return copy.deepcopy(routes_list)

    new_routes = copy.deepcopy(routes_list)
    r1_idx, r2_idx = random.sample(range(len(new_routes)), 2)
    route1, route2 = new_routes[r1_idx], new_routes[r2_idx]

    if len(route1) < n + 2 or len(route2) < n + 2:
        return new_routes

    s1 = random.randint(1, len(route1) - n - 1)
    s2 = random.randint(1, len(route2) - n - 1)

    seg1 = route1[s1 : s1 + n]
    seg2 = route2[s2 : s2 + n]

    new_routes[r1_idx] = route1[:s1] + seg2 + route1[s1 + n :]
    new_routes[r2_idx] = route2[:s2] + seg1 + route2[s2 + n :]

    return new_routes


def remove_n_bins_consecutive(routes_list: list, removed_bins: set, bins_cannot_removed: set, n: int = 2) -> list:
    """
    Remove a consecutive sequence of bins from a route.
    """
    new_routes = copy.deepcopy(routes_list)
    valid_routes = [i for i, r in enumerate(new_routes) if len(r) >= n + 2]
    if not valid_routes:
        return new_routes

    r_idx = random.choice(valid_routes)
    route = new_routes[r_idx]

    # Try to find a segment without 'must-go' bins
    start_indices = list(range(1, len(route) - n))
    random.shuffle(start_indices)

    for start in start_indices:
        segment = route[start : start + n]
        if not any(b in bins_cannot_removed for b in segment):
            for b in segment:
                removed_bins.add(b)
            new_routes[r_idx] = route[:start] + route[start + n :]
            break

    return new_routes


def add_n_bins_consecutive(
    routes_list: list,
    removed_bins: set,
    stocks: dict,
    vehicle_capacity: float,
    id_to_index: dict,
    distance_matrix,
    n: int = 2,
) -> list:
    """
    Add a sequence of bins from removed set as a consecutive segment.
    """
    new_routes = copy.deepcopy(routes_list)
    if len(removed_bins) < n or not new_routes:
        return new_routes

    bins_to_add = random.sample(list(removed_bins), n)
    total_weight = sum(stocks.get(b, 0) for b in bins_to_add)

    r_idx = random.randint(0, len(new_routes) - 1)
    route = new_routes[r_idx]
    load = sum(stocks.get(x, 0) for x in route if x != 0)

    if load + total_weight <= vehicle_capacity:
        # Insert them consecutively at a random position
        pos = random.randint(1, len(route) - 1)
        new_routes[r_idx] = route[:pos] + bins_to_add + route[pos:]
        for b in bins_to_add:
            removed_bins.discard(b)

    return new_routes


def add_route_with_removed_bins_consecutive(
    routes_list: list, removed_bins: set, stocks: dict, vehicle_capacity: float
) -> list:
    """
    Alias for random version as 'consecutive' doesn't strictly apply to a sequence
    created from a pool, but we maintain the interface.
    """
    return add_route_with_removed_bins_random(routes_list, removed_bins, stocks, vehicle_capacity)
