"""
Adaptive mutation and neighborhood operators for SANS-style optimization.

Implements specialized local search operators including cross-exchange,
relocation, Or-opt moves, and 2-opt swaps tailored for the SANS (Simulated
Annealing with Adaptive Neighborhood Selection) policy variant.

This module acts as a facade, re-exporting operators from specialized sub-modules.
"""

from .sans_neighborhoods import (
    cross_exchange,
    get_2opt_neighbors,
    get_neighbors,
    insert_bin_in_route,
    move_between_routes,
    mutate_route_by_swapping_bins,
    or_opt_move,
    relocate_within_route,
)
from .sans_perturbations import (
    add_n_bins_consecutive,
    add_n_bins_random,
    add_route_with_removed_bins_consecutive,
    add_route_with_removed_bins_random,
    move_n_route_consecutive,
    move_n_route_random,
    remove_bins_from_route,
    remove_n_bins_consecutive,
    remove_n_bins_random,
    swap_n_route_consecutive,
    swap_n_route_random,
)

__all__ = [
    "get_neighbors",
    "get_2opt_neighbors",
    "relocate_within_route",
    "cross_exchange",
    "or_opt_move",
    "move_between_routes",
    "insert_bin_in_route",
    "mutate_route_by_swapping_bins",
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
