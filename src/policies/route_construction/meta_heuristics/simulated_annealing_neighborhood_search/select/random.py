"""
Random selection strategies for bin management.
"""

from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.routes import (
    organize_route,
)


# Function to remove one bin from one route (Drop one bin from one random chosen route)
def remove_bin(routes_list, removed_bins, bins_cannot_removed, rng):
    """
    Remove a random bin from one of the routes and add it to the removed set.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (List[int]): Global set of uncollected bins.
        bins_cannot_removed (List[int]): Mandatory bins.
        rng (random.Random): Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Updated solution and removed set.
    """
    bin_to_remove = None
    if len(routes_list) > 0:
        chosen_route = rng.sample(routes_list, 1)[0]
        if len(chosen_route) > len(bins_cannot_removed) + 2:
            if len(bins_cannot_removed) > 0:
                bin_to_remove = rng.sample(bins_cannot_removed, 1)[0]
                i = 0
                while bin_to_remove in bins_cannot_removed:  # and i <= len(chosen_route):
                    bin_to_remove = rng.sample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                    i += 1
            else:
                bin_to_remove = rng.sample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

            removed_bins.append(bin_to_remove)
            chosen_route.remove(bin_to_remove)
        if len(chosen_route) == 2:
            routes_list.remove(chosen_route)

        routes_list = list(filter(None, routes_list))
    return bin_to_remove


# Function to add bin from the removed bins set to one route
# (Add one bin from the removed bins set to one random chosen route)
def add_bin(routes_list, removed_bins, rng):
    """
    Pick a random bin from the removed set and add it to a random route.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (List[int]): Set of available removed bins.
        rng (random.Random): Random number generator.

    Returns:
        List[List[int]]: Mutated routing solution.
    """
    if len(routes_list) > 0 and len(removed_bins) > 0:
        bin_to_add = rng.sample(removed_bins, 1)[0]
        chosen_route = rng.sample(routes_list, 1)[0]
        chosen_position = chosen_route.index(rng.sample(chosen_route[1 : len(chosen_route) - 1], 1)[0])

        chosen_route.insert(chosen_position, bin_to_add)
        removed_bins.remove(bin_to_add)
        return bin_to_add


# Function to remove n bin from route randomly
def remove_n_bins_random(routes_list, removed_bins, bins_cannot_removed, rng):
    """
    Remove n random bins from the routing solution.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (List[int]): Target set for removed bins.
        bins_cannot_removed (List[int]): Mandatory bins.
        rng (random.Random): Random number generator.

    Returns:
        List[List[int]]: Routing solution with fewer bins.
    """
    bins_to_remove_random = []
    possible_n = [2, 3, 4, 5]
    chosen_n = rng.sample(possible_n, 1)[0]
    length = 0
    for i in routes_list:
        length = length + len(i)

    n_routes = len(routes_list)
    bin_to_remove = None
    if n_routes > 0:
        chosen_route = rng.sample(routes_list, 1)[0]
        if len(chosen_route) - 2 - len(bins_cannot_removed) > chosen_n:
            for i in range(0, chosen_n):
                if len(bins_cannot_removed) > 0:
                    bin_to_remove = rng.sample(bins_cannot_removed, 1)[0]
                    i = 0
                    while bin_to_remove in bins_cannot_removed and i <= 100:
                        bin_to_remove = rng.sample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                        i += 1
                else:
                    bin_to_remove = rng.sample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                removed_bins.append(bin_to_remove)
                chosen_route.remove(bin_to_remove)
                bins_to_remove_random.append(bin_to_remove)
            if len(chosen_route) == 2:
                routes_list.remove(chosen_route)
        routes_list = list(filter(None, routes_list))
    return bins_to_remove_random, chosen_n


# Function to add n bin from the removed bins set to any route randomly
def add_n_bins_random(routes_list, removed_bins, rng):
    """
    Add n random bins from the removed set to random routes.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (List[int]): Set of removed bins.
        rng (random.Random): Random number generator.

    Returns:
        Tuple[List[int], int]: List of added bins and the number n chosen.
    """
    bins_to_add_random = []
    possible_n = [2, 3, 4, 5]
    chosen_n = rng.sample(possible_n, 1)[0]
    if len(routes_list) > 0 and len(removed_bins) >= chosen_n:
        for _b in range(0, chosen_n):
            bin_to_add = rng.sample(removed_bins, 1)[0]
            chosen_route = rng.sample(routes_list, 1)[0]
            chosen_position = chosen_route.index(rng.sample(chosen_route[1 : len(chosen_route) - 1], 1)[0])

            chosen_route.insert(chosen_position, bin_to_add)
            removed_bins.remove(bin_to_add)
            bins_to_add_random.append(bin_to_add)
    return bins_to_add_random, chosen_n


# Add one route random
def add_route_random(routes_list, distance_matrix, rng):
    """
    Add a new route with a single bin chosen randomly from existing routes.

    Args:
        routes_list (List[List[int]]): All routes.
        distance_matrix (np.ndarray): For sequencing.
        rng (random.Random): Random number generator.

    Returns:
        List[List[int]]: Updated set of routes.
    """
    if len(routes_list) > 0:
        chosen_route = rng.sample(routes_list, 1)[0]
    else:
        return routes_list

    length_chosen_route = len(chosen_route)
    possible_percent = [0.3, 0.4, 0.5, 0.6]
    chosen_n = rng.sample(possible_percent, 1)[0]
    chosen_n_percent = int(chosen_n * length_chosen_route)
    bins = []
    for _s in range(0, chosen_n_percent):
        chosen_bin = rng.sample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
        chosen_route.remove(chosen_bin)
        bins.append(chosen_bin)

    route = organize_route(bins, distance_matrix)
    routes_list.append(route)
    if len(chosen_route) == 2:
        chosen_route = []

    routes_list = list(filter(None, routes_list))
    return chosen_n


# Add one route with bins from removed bins
def add_route_with_removed_bins_random(routes_list, removed_bins, distance_matrix, rng):
    """
    Create a new route using a random subset of removed bins.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (List[int]): Pool of removed bins.
        distance_matrix (np.ndarray): Distance matrix.
        rng (random.Random): Random number generator.

    Returns:
        Tuple[int, List[int]]: Number of bins used and the list of bins used.
    """
    length_removed_bins = len(removed_bins)
    possible_percent = [0.2, 0.3, 0.4, 0.5]
    chosen_n = rng.sample(possible_percent, 1)[0]
    chosen_n_percent = int(chosen_n * length_removed_bins)
    bins_random = []
    for _o in range(0, chosen_n_percent):
        chosen_bin = rng.sample(removed_bins, 1)[0]
        removed_bins.remove(chosen_bin)
        bins_random.append(chosen_bin)

    chosen_bins = bins_random.copy()
    route = organize_route(chosen_bins, distance_matrix)
    routes_list.append(route)
    return chosen_n, bins_random
