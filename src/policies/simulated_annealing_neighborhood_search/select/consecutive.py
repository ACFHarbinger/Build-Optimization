"""
Consecutive selection strategies for bin management.
"""

from random import sample as rsample

from logic.src.policies.simulated_annealing_neighborhood_search.common.routes import organize_route


def _extract_valid_segment(chosen_route, chosen_n, bins_cannot_removed):
    """Attempt to find a valid segment of length chosen_n to remove."""
    # Try finding a valid segment up to 100 times
    # Note: original loop just tried 100 times randomly.
    # A more deterministic approach would be sliding window, but let's stick to the stochastic nature.

    if len(chosen_route) - 2 < chosen_n:
        return []

    for _ in range(100):
        # start index range: 1 to len - 1 - chosen_n
        max_start = len(chosen_route) - 1 - chosen_n
        if max_start < 1:
            break

        start_idx = rsample(range(1, max_start + 1), 1)[0]  # rsample returns list

        # Check if segment contains mandatory bins
        segment = chosen_route[start_idx : start_idx + chosen_n]
        if not any(b in bins_cannot_removed for b in segment):
            return segment

    return []


def remove_n_bins_consecutive(routes_list, removed_bins, bins_cannot_removed):
    """
    Remove n consecutive bins from a route.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (List[int]): Target set.
        bins_cannot_removed (List[int]): Mandatory nodes.

    Returns:
        List[List[int]]: Mutated routing solution.
    """
    bins_to_remove_consecutive = []  # type: ignore[var-annotated]
    possible_n = [2, 3, 4, 5]
    chosen_n = rsample(possible_n, 1)[0]

    if not routes_list:
        return bins_to_remove_consecutive, chosen_n

    chosen_route = rsample(routes_list, 1)[0]

    # Calculate if possible to remove n

    # Check if route is long enough and feasible
    # Optimization: count mandatory bins in route
    mandatory_in_route = sum(1 for b in chosen_route if b in bins_cannot_removed)
    available_bins = len(chosen_route) - 2  # excluding depots

    if available_bins - mandatory_in_route < chosen_n:
        # Not enough removable bins total, certainly no consecutive segment of n
        return bins_to_remove_consecutive, chosen_n

    if not bins_cannot_removed:
        # Fast path: just pick any segment
        if available_bins >= chosen_n:
            max_start = len(chosen_route) - 1 - chosen_n
            start_idx = rsample(range(1, max_start + 1), 1)[0]
            segment = chosen_route[start_idx : start_idx + chosen_n]
            bins_to_remove_consecutive.extend(segment)
            removed_bins.extend(segment)
            del chosen_route[start_idx : start_idx + chosen_n]
    else:
        # Complex path with mandatory constraints
        segment = _extract_valid_segment(chosen_route, chosen_n, bins_cannot_removed)
        if segment:
            bins_to_remove_consecutive.extend(segment)
            removed_bins.extend(segment)

            # To delete efficiently, we need index.
            if len(segment) > 0:
                first_bin_idx = chosen_route.index(segment[0])
                del chosen_route[first_bin_idx : first_bin_idx + len(segment)]

    if len(chosen_route) <= 2 and chosen_route in routes_list:
        routes_list.remove(chosen_route)

    # Clean up empty routes
    routes_list[:] = [r for r in routes_list if len(r) > 2]

    return bins_to_remove_consecutive, chosen_n


# Function to add n bin from the removed bins set to any route randomly
def add_n_bins_consecutive(routes_list, removed_bins):
    """
    Add n consecutive bins from the removed set (treated as list) to a random route.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (List[int]): Set/List of removed bins.

    Returns:
        Tuple[List[int], int]: List of added bins and the number n chosen.
    """
    bins_to_add_consecutive = []
    possible_n = [2, 3, 4, 5]
    chosen_n = rsample(possible_n, 1)[0]
    if len(routes_list) > 0 and len(removed_bins) >= chosen_n:
        for _b in range(0, chosen_n):
            bin_to_add = rsample(removed_bins, 1)[0]
            removed_bins.remove(bin_to_add)
            bins_to_add_consecutive.append(bin_to_add)

        chosen_route = rsample(routes_list, 1)[0]
        chosen_position = chosen_route.index(rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0])
        for d in range(0, chosen_n):
            chosen_route.insert(chosen_position + d, bins_to_add_consecutive[d])
    return bins_to_add_consecutive, chosen_n


# Add one route consecutive
def add_route_consecutive(routes_list, distance_matrix):
    """
    Create a new route by extracting a consecutive segment from an existing route.

    Args:
        routes_list (List[List[int]]): Current routes.
        distance_matrix (np.ndarray): Distance matrix for route organization.

    Returns:
        int: The number of bins moved to the new route.
    """
    if len(routes_list) > 0:
        chosen_route = rsample(routes_list, 1)[0]
    else:
        return 0

    length_chosen_route = len(chosen_route)
    possible_percent = [0.3, 0.4, 0.5, 0.6]
    chosen_n = rsample(possible_percent, 1)[0]
    chosen_n_percent = int(chosen_n * length_chosen_route)
    bins = []

    position_chosen_bin = 0
    for s in range(0, chosen_n_percent):
        if s == 0:
            chosen_bin = rsample(chosen_route[1 : len(chosen_route) - chosen_n_percent], 1)[0]
            position_chosen_bin = chosen_route.index(chosen_bin)
        else:
            chosen_bin = chosen_route[position_chosen_bin + s]

        bins.append(chosen_bin)

    chosen_bins = bins.copy()
    route = organize_route(bins, distance_matrix)
    routes_list.append(route)
    for u in chosen_bins:
        chosen_route.remove(u)

    if len(chosen_route) == 2:
        chosen_route = []

    routes_list = list(filter(None, routes_list))
    return chosen_n


# Add one route with bins from removed bins consecutive
def add_route_with_removed_bins_consecutive(routes_list, removed_bins, distance_matrix):
    """
    Create a new route using a consecutive sequence of removed bins.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (List[int]): Pool of removed bins.
        distance_matrix (np.ndarray): Distance matrix.

    Returns:
        Tuple[int, List[int]]: Number of bins used and the list of bins used.
    """
    length_removed_bins = len(removed_bins)
    possible_percent = [0.2, 0.3, 0.4, 0.5]
    chosen_n = rsample(possible_percent, 1)[0]
    chosen_n_percent = int(chosen_n * length_removed_bins)
    bins_consecutive = []

    position_chosen_bin = 0
    for o in range(0, chosen_n_percent):
        if o == 0:
            chosen_bin = rsample(removed_bins[0 : len(removed_bins) - chosen_n_percent + 1], 1)[0]
            position_chosen_bin = removed_bins.index(chosen_bin)
        else:
            chosen_bin = removed_bins[position_chosen_bin + o]

        bins_consecutive.append(chosen_bin)

    chosen_bins = bins_consecutive.copy()
    chosen_bins_organize = bins_consecutive.copy()
    route = organize_route(chosen_bins_organize, distance_matrix)
    routes_list.append(route)
    for u in chosen_bins:
        removed_bins.remove(u)
    return chosen_n, bins_consecutive
