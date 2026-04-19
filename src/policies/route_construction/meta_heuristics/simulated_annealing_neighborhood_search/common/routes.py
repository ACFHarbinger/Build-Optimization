import random
from copy import deepcopy
from typing import List

import numpy as np
from shapely.geometry import LineString

from .distance import compute_sans_route_cost
from .objectives import compute_profit, compute_real_profit


def create_points(data, bins_coordinates):
    """
    Create a mapping of bin IDs to their coordinates.

    Args:
        data (pd.DataFrame): Bin metadata containing '#bin'.
        bins_coordinates (List[Tuple[float, float]]): List of coordinate pairs.

    Returns:
        Dict[int, Tuple[float, float]]: Mapping from bin ID to coordinates.
    """
    points = []
    pair_lng_lat = []  # type: ignore[var-annotated]
    for index, _row in data.iterrows():
        lat = bins_coordinates["Lat"][index]
        lng = bins_coordinates["Lng"][index]
        pair_lng_lat = []
        pair_lng_lat.append(lng)
        pair_lng_lat.append(lat)
        points.append(pair_lng_lat)
    return points


def _find_crossed_arcs(route, points, route_idx, cache):
    """Identify intersecting arcs within a route."""
    arc_pair = []
    arc1 = []
    arc2 = []
    cache[route_idx] = []

    # Iterate through edges (arcs)
    n = len(route)

    for i in range(n - 3):
        I_1 = route[i]
        F_1 = route[i + 1]
        line1 = LineString([(points[I_1][0], points[I_1][1]), (points[F_1][0], points[F_1][1])])

        for j in range(i + 2, n - 1):
            I_2 = route[j]
            F_2 = route[j + 1]
            line2 = LineString([(points[I_2][0], points[I_2][1]), (points[F_2][0], points[F_2][1])])

            if line1.intersects(line2):
                arc1 = [I_1, F_1]
                arc2 = [I_2, F_2]
                arc_pair = [arc1, arc2]
                cache[route_idx].append(arc_pair)


def _remove_invalid_crossings(crossings, points):
    """Filter out invalid crossings (depot-connected or sharing endpoints)."""
    valid_crossings = []
    for pair in crossings:
        # Check if depot (0) is involved
        if 0 in pair[0] and 0 in pair[1]:
            continue

        # Check if endpoints are shared (geometrically identical points)
        # pair[0][1] is F_1, pair[1][0] is I_2
        p1 = points[pair[0][1]]
        p2 = points[pair[1][0]]
        if p1[0] == p2[0] and p1[1] == p2[1]:
            continue
        valid_crossings.append(pair)
    return valid_crossings


def uncross_arcs_in_routes(
    previous_solution,
    p_vehicle,
    p_load,
    p_route_difference,
    p_shift,
    data,
    points,
    distance_matrix,
    values,
):
    """
    Improve routes by uncrossing arcs while maintaining profit constraints.

    Uses geometric LineString intersections to identify crossed arcs.

    Args:
        previous_solution (List[List[int]]): Solution to improve.
        p_vehicle (float): Vehicle penalty.
        p_load (float): Load penalty.
        p_route_difference (float): Difference penalty.
        p_shift (float): Shift penalty.
        data (pd.DataFrame): Bin weights data.
        points (Dict): Bin coordinates.
        distance_matrix (np.ndarray): Distance matrix.
        values (Dict): Parameter dictionary.

    Returns:
        List[List[int]]: Improved routing solution.
    """
    # Local cache instead of globals
    crossed_arcs_cache = {}  # type: ignore[var-annotated]

    # 1. Identify initial crossings
    for idx, route in enumerate(previous_solution):
        _find_crossed_arcs(route, points, idx, crossed_arcs_cache)

    # 2. Process and resolve crossings
    for idx, route in enumerate(previous_solution):
        # Retrieve crossings for this route
        current_crossings = crossed_arcs_cache.get(idx, [])
        relevant_crossings = _remove_invalid_crossings(current_crossings, points)

        while relevant_crossings:
            # Perform swap on the first valid crossing
            first_crossing = relevant_crossings[0]
            swap_bin_1 = first_crossing[0][1]
            swap_bin_2 = first_crossing[1][0]

            try:
                pos1 = route.index(swap_bin_1)
                pos2 = route.index(swap_bin_2)
            except ValueError:
                # Bins might have been moved/removed? Should not happen in this logic if consistent
                relevant_crossings.pop(0)
                continue

            # Apply 2-opt swap
            # Swap endpoints
            route[pos1], route[pos2] = route[pos2], route[pos1]
            # Reverse segment in between
            if pos1 + 1 < pos2:
                route[pos1 + 1 : pos2] = route[pos1 + 1 : pos2][::-1]

            # Re-evaluate crossings for this route after modification
            _find_crossed_arcs(route, points, idx, crossed_arcs_cache)
            current_crossings = crossed_arcs_cache.get(idx, [])
            relevant_crossings = _remove_invalid_crossings(current_crossings, points)

    solution_after_uncross = deepcopy(previous_solution)
    profit_after_uncross = compute_profit(
        solution_after_uncross,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )
    uncross_profit = compute_real_profit(solution_after_uncross, p_vehicle, data, distance_matrix, values)
    return solution_after_uncross, profit_after_uncross, uncross_profit


def rearrange_part_route(
    routes_list: List[List[int]], distance_matrix: np.ndarray, rng: random.Random
) -> List[List[int]]:
    """
    Select a random portion of a route and reorder it using a greedy Nearest Neighbor heuristic.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        distance_matrix (np.ndarray): Distance matrix.
        rng (random.Random): Random number generator.

    Returns:
        List[List[int]]: Routing solution with partially reordered route.
    """
    if not routes_list:
        return [[0]]

    chosen_route: List[int] = rng.sample(routes_list, 1)[0]
    length_chosen_route = len(chosen_route)

    if length_chosen_route < 4:
        return [[0]]

    possible_percent = [0.1, 0.2, 0.3, 0.4]
    chosen_n = rng.sample(possible_percent, 1)[0]
    chosen_n_percent = int(chosen_n * length_chosen_route)

    if chosen_n_percent < 2:
        return [[0]]

    bins = []

    max_start_index = length_chosen_route - 1 - chosen_n_percent
    if max_start_index < 1:
        return [[0]]

    start_index = rng.randint(1, max_start_index)
    segment = chosen_route[start_index : start_index + chosen_n_percent]

    # Remove segment from route
    bins = segment
    del chosen_route[start_index : start_index + chosen_n_percent]

    # Reorder bins using NN
    sorted_route_full = organize_route(bins, distance_matrix)
    # Remove depots (starts 0, ends 0)
    sorted_segment = sorted_route_full[1:-1]

    # Insert back at the same position
    for i, bin_id in enumerate(sorted_segment):
        chosen_route.insert(start_index + i, bin_id)

    routes_list[routes_list.index(chosen_route)] = chosen_route
    return routes_list


# Function to generate a route using the nearest neighbor (starting from the left)
def organize_route(bins, distance_matrix):
    """
    Construct a route from a set of bins using the Nearest Neighbor heuristic.

    Starts from the first bin (depot) and iteratively adds the closest unvisited bin.

    Args:
        bins (List[int]): Set of bin IDs to sequence.
        distance_matrix (np.ndarray): Distance matrix.

    Returns:
        List[int]: Sequenced route starting and ending at the depot.
    """
    depot = 0
    route = []
    distances = []
    bins_sequence = []

    # Choose depot to initialize the route
    bin_chosen = depot
    route.append(bin_chosen)

    # Previous_bin is the bin previously added in the route
    previous_bin = bin_chosen
    while len(bins) != 0:
        for i in bins:
            dist = distance_matrix[previous_bin][i]
            distances.append(dist)
            bins_sequence.append(i)

        min_distance = min(distances)
        distance_index = distances.index(min_distance)
        bin_chosen = bins_sequence[distance_index]
        route.append(bin_chosen)
        previous_bin = bin_chosen

        bins.remove(previous_bin)
        distances = []
        bins_sequence = []
    route.append(depot)
    return route


# Lookahead sans policy functions
def two_opt_uncross_arc(route, distance_matrix, id_to_index):
    """
    Apply 2-opt local search to a single route to remove arc crossings.

    Args:
        route (List[int]): Node sequence.
        distance_matrix (np.ndarray): Distance matrix.
        id_to_index (Dict[int, int]): Node ID to matrix index mapping.

    Returns:
        List[int]: Optimized route.
    """
    best = route
    improved = True
    while improved:
        improved = False
        best_distance = compute_sans_route_cost(best, distance_matrix, id_to_index)
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                new_route = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                new_distance = compute_sans_route_cost(new_route, distance_matrix, id_to_index)
                if new_distance < best_distance:
                    best = new_route
                    improved = True
                    break
            if improved:
                break
    return best


def uncross_arcs_in_sans_routes(routes, id_to_index, distance_matrix):
    """
    Apply 2-opt uncrossing to all routes in a solution.

    Args:
        routes (List[List[int]]): Solution to optimize.
        id_to_index (Dict[int, int]): Node mapping.
        distance_matrix (np.ndarray): Distance matrix.

    Returns:
        List[List[int]]: Improved solution.
    """
    cleaned_routes = [[int(x) for x in route] for route in routes]
    new_routes = [
        (two_opt_uncross_arc(route, distance_matrix, id_to_index) if len(route) > 3 else route)
        for route in cleaned_routes
    ]
    return new_routes
