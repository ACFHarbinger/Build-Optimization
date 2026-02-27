"""
Inter-route swap operators for the Look-Ahead policy.
"""

from random import sample as rsample


def swap_2_routes(routes_list):
    """
    Inter-route perturbation: Swap one bin from one route with another bin
    from a different route.

    Args:
        routes_list (List[List[int]]): Current routing solution.

    Returns:
        List[List[int]]: Mutated routing solution.
    """
    if len(routes_list) > 1:
        chosen_routes = rsample(routes_list, 2)
        route1, route2 = chosen_routes

        if len(route1) > 2 and len(route2) > 2:
            bin1 = rsample(route1[1 : len(route1) - 1], 1)[0]
            bin2 = rsample(route2[1 : len(route2) - 1], 1)[0]

            pos1 = route1.index(bin1)
            pos2 = route2.index(bin2)

            route1[pos1], route2[pos2] = route2[pos2], route1[pos1]


def swap_n_2_routes_random(routes_list):
    """
    Inter-route perturbation: Swap n random bins between two different routes.

    Args:
        routes_list (List[List[int]]): Current routing solution.

    Returns:
        int: The number of bins swapped (chosen_n).
    """
    chosen_n = rsample([2, 3, 4, 5], 1)[0]

    if len(routes_list) > 1:
        chosen_routes = rsample(routes_list, 2)
        route1, route2 = chosen_routes

        if len(route1) >= chosen_n + 2 and len(route2) >= chosen_n + 2:
            for _ in range(chosen_n):
                bin1 = rsample(route1[1 : len(route1) - 1], 1)[0]
                bin2 = rsample(route2[1 : len(route2) - 1], 1)[0]

                pos1 = route1.index(bin1)
                pos2 = route2.index(bin2)

                route1[pos1], route2[pos2] = route2[pos2], route1[pos1]

    return chosen_n


def swap_n_2_routes_consecutive(routes_list):
    """
    Inter-route perturbation: Swap two consecutive sequences of n bins
    between two different routes.

    Args:
        routes_list (List[List[int]]): Current routing solution.

    Returns:
        int: The number of consecutive nodes swapped (chosen_n).
    """
    chosen_n = rsample([2, 3, 4, 5], 1)[0]

    if len(routes_list) > 1:
        chosen_routes = rsample(routes_list, 2)
        route1, route2 = chosen_routes

        if len(route1) >= chosen_n + 2 and len(route2) >= chosen_n + 2:
            # Pick start indices for both segments
            start1 = rsample(range(1, len(route1) - chosen_n), 1)[0]
            start2 = rsample(range(1, len(route2) - chosen_n), 1)[0]

            # Swap segments
            seg1 = route1[start1 : start1 + chosen_n]
            seg2 = route2[start2 : start2 + chosen_n]

            route1[start1 : start1 + chosen_n] = seg2
            route2[start2 : start2 + chosen_n] = seg1

    return chosen_n
