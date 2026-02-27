"""
Randomized local search strategy.
"""

import numpy as np

from logic.src.policies.simulated_annealing_neighborhood_search.common.routes import rearrange_part_route
from logic.src.policies.simulated_annealing_neighborhood_search.operators.move import (
    move_1_route,
    move_2_routes,
    move_n_2_routes_consecutive,
    move_n_2_routes_random,
    move_n_route_consecutive,
    move_n_route_random,
)
from logic.src.policies.simulated_annealing_neighborhood_search.operators.swap import (
    swap_1_route,
    swap_2_routes,
    swap_n_2_routes_consecutive,
    swap_n_2_routes_random,
    swap_n_route_consecutive,
    swap_n_route_random,
)
from logic.src.policies.simulated_annealing_neighborhood_search.select import (
    add_bin,
    add_n_bins_consecutive,
    add_n_bins_random,
    add_route_consecutive,
    add_route_random,
    add_route_with_removed_bins_consecutive,
    add_route_with_removed_bins_random,
    remove_bin,
    remove_n_bins_consecutive,
    remove_n_bins_random,
)


def local_search(routes_list, removed_bins, distance_matrix, bins_cannot_removed):  # noqa: C901
    """
    Perform a multi-operator local search on the routing solution.

    Randomly selects move and swap operators to explore the neighborhood.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        removed_bins (List[int]): Set of bins not currently collected.
        distance_matrix (np.ndarray): Distance matrix.
        bins_cannot_removed (List[int]): Bins that must remain in the routes.

    Returns:
        List[List[int]]: Mutated routing solution.
    """
    procedures = [
        "Move 1 route",
        "Swap 1 route",
        "Move 2 routes",
        "Swap 2 routes",
        "Drop bin",
        "Add bin",
        "Move n 1 route random",
        "Move n 1 route consecutive",
        "Swap n 1 route random",
        "Swap n 1 route consecutive",
        "Move n 2 routes random",
        "Move n 2 routes consecutive",
        "Swap n 2 routes random",
        "Swap n 2 routes consecutive",
        "Remove n bins random",
        "Remove n bins consecutive",
        "Add n bins random",
        "Add n bins consecutive",
        "Add route random",
        "Add route consecutive",
        "Add route with removed bins random",
        "Add route with removed bins consecutive",
        "Rearrange part of 1 route",
    ]  #'Remove bins' #'Insert bins'

    chosen_procedure = np.random.choice(
        procedures,
        1,
        p=[
            1 / 19 + 4.440892098500626e-16,
            1 / 19,
            1 / 19,
            1 / 19,
            1 / 19,
            1 / 19,
            1 / 19,
            1 / 19,
            1 / 19,
            1 / 19,
            1 / 19,
            1 / 19,
            1 / 19,
            1 / 19,
            1 / 19,
            1 / 19,
            1 / 19,
            1 / 19,
            1 / 19,
            0,
            0,
            0,
            0,
            1 / 19,
        ],
    )[0]
    bin_to_remove = None
    bin_to_add = None
    bins_to_remove_random = []
    bins_to_remove_consecutive = []
    bins_to_add_random = []
    bins_to_add_consecutive = []
    bins_random = []
    bins_consecutive = []
    chosen_n = 1
    if chosen_procedure.startswith("Move") or chosen_procedure.startswith("Swap"):
        if "n" not in chosen_procedure:
            # simple move/swap
            if chosen_procedure == "Move 1 route":
                move_1_route(routes_list)
            elif chosen_procedure == "Swap 1 route":
                swap_1_route(routes_list)
            elif chosen_procedure == "Move 2 routes":
                move_2_routes(routes_list)
            elif chosen_procedure == "Swap 2 routes":
                swap_2_routes(routes_list)
        else:
            # n move/swap
            if chosen_procedure == "Move n 1 route random":
                chosen_n = move_n_route_random(routes_list)  # type: ignore[assignment]
            elif chosen_procedure == "Move n 1 route consecutive":
                chosen_n = move_n_route_consecutive(routes_list)  # type: ignore[assignment]
            elif chosen_procedure == "Swap n 1 route random":
                chosen_n = swap_n_route_random(routes_list)
            elif chosen_procedure == "Swap n 1 route consecutive":
                chosen_n = swap_n_route_consecutive(routes_list)
            elif chosen_procedure == "Move n 2 routes random":
                chosen_n = move_n_2_routes_random(routes_list)  # type: ignore[assignment]
            elif chosen_procedure == "Move n 2 routes consecutive":
                chosen_n = move_n_2_routes_consecutive(routes_list)  # type: ignore[assignment]
            elif chosen_procedure == "Swap n 2 routes random":
                chosen_n = swap_n_2_routes_random(routes_list)
            elif chosen_procedure == "Swap n 2 routes consecutive":
                chosen_n = swap_n_2_routes_consecutive(routes_list)

    elif "Remove n bins" in chosen_procedure:
        if "random" in chosen_procedure:
            bins_to_remove_random, chosen_n = remove_n_bins_random(routes_list, removed_bins, bins_cannot_removed)
        else:
            bins_to_remove_consecutive, chosen_n = remove_n_bins_consecutive(
                routes_list, removed_bins, bins_cannot_removed
            )

    elif "Add n bins" in chosen_procedure:
        if "random" in chosen_procedure:
            bins_to_add_random, chosen_n = add_n_bins_random(routes_list, removed_bins)
        else:
            bins_to_add_consecutive, chosen_n = add_n_bins_consecutive(routes_list, removed_bins)

    elif "Add route" in chosen_procedure:
        if "removed bins" in chosen_procedure:
            if "random" in chosen_procedure:
                chosen_n, bins_random = add_route_with_removed_bins_random(routes_list, removed_bins, distance_matrix)
            else:
                chosen_n, bins_consecutive = add_route_with_removed_bins_consecutive(
                    routes_list, removed_bins, distance_matrix
                )
        else:
            if "random" in chosen_procedure:
                chosen_n = add_route_random(routes_list, distance_matrix)
            else:
                chosen_n = add_route_consecutive(routes_list, distance_matrix)

    elif chosen_procedure == "Drop bin":
        bin_to_remove = remove_bin(routes_list, removed_bins, bins_cannot_removed)
    elif chosen_procedure == "Add bin":
        bin_to_add = add_bin(routes_list, removed_bins)
    elif chosen_procedure == "Rearrange part of 1 route":
        chosen_n = rearrange_part_route(routes_list, distance_matrix)
    return (
        chosen_procedure,
        chosen_n,
        bin_to_remove,
        bin_to_add,
        bins_to_remove_random,
        bins_to_remove_consecutive,
        bins_to_add_random,
        bins_to_add_consecutive,
        bins_random,
        bins_consecutive,
    )
