"""
Operator application logic for Simulated Annealing.
"""

import random

from .sans_neighborhoods import (
    cross_exchange,
    get_2opt_neighbors,
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
    remove_n_bins_consecutive,
    remove_n_bins_random,
    swap_n_route_consecutive,
    swap_n_route_random,
)


def _handle_2opt(solution):
    """Apply 2-opt operator."""
    valid_indices = [i for i, r in enumerate(solution) if len(r) > 3]
    if valid_indices:
        r = random.choice(valid_indices)
        solution[r] = random.choice(get_2opt_neighbors(solution[r]))
    return solution


def _handle_move(solution, data, vehicle_capacity, id_to_index):
    """Apply move operator."""
    neighbors = move_between_routes(solution, data, vehicle_capacity, id_to_index)
    if neighbors:
        return random.choice(neighbors)
    return solution


def _handle_swap(solution):
    """Apply swap operator."""
    r = random.choice(range(len(solution)))
    solution[r] = mutate_route_by_swapping_bins(solution[r], num_bins=random.choice([1, 2]))
    return solution


def _handle_insert(solution, data, stocks, vehicle_capacity, id_to_index, distance_matrix, candidate_removed_bins):
    """Apply insert operator."""
    if not solution:
        return solution
    r = random.choice(range(len(solution)))
    all_bins = set(data["#bin"]) - {0}
    used_bins = set(b for route in solution for b in route)
    unused = list(all_bins - used_bins)
    if unused:
        bin_to_insert = random.choice(unused)
        load = sum(stocks.get(b, 0) for b in solution[r] if b != 0)
        if load + stocks.get(bin_to_insert, 0) <= vehicle_capacity:
            solution[r] = insert_bin_in_route(solution[r], bin_to_insert, id_to_index, distance_matrix)
            if bin_to_insert in candidate_removed_bins:
                candidate_removed_bins.remove(bin_to_insert)
    return solution


def apply_operator(
    op,
    new_solution,
    candidate_removed_bins,
    data,
    vehicle_capacity,
    id_to_index,
    stocks,
    must_go_bins,
    distance_matrix,
):
    """
    Apply the selected operator to the solution.
    Modifies new_solution in place or returns a new one.
    """
    # Simple operators using direct function calls or simple lambdas
    handlers = {
        "2opt": lambda sol: _handle_2opt(sol),
        "move": lambda sol: _handle_move(sol, data, vehicle_capacity, id_to_index),
        "swap": lambda sol: _handle_swap(sol),
        "remove": lambda sol: remove_n_bins_random(sol, candidate_removed_bins, must_go_bins, n=1),
        "move_n_random": lambda sol: move_n_route_random(sol, n=random.randint(2, 5)),
        "move_n_consec": lambda sol: move_n_route_consecutive(sol, n=random.randint(2, 5)),
        "swap_n_random": lambda sol: swap_n_route_random(sol, n=random.randint(2, 5)),
        "swap_n_consec": lambda sol: swap_n_route_consecutive(sol, n=random.randint(2, 5)),
        "remove_n_bins": lambda sol: remove_n_bins_random(
            sol, candidate_removed_bins, must_go_bins, n=random.randint(2, 5)
        ),
        "remove_n_bins_consec": lambda sol: remove_n_bins_consecutive(
            sol, candidate_removed_bins, must_go_bins, n=random.randint(2, 5)
        ),
        "add_n_bins": lambda sol: add_n_bins_random(
            sol, candidate_removed_bins, stocks, vehicle_capacity, id_to_index, distance_matrix, n=2
        ),
        "add_n_bins_consec": lambda sol: add_n_bins_consecutive(
            sol, candidate_removed_bins, stocks, vehicle_capacity, id_to_index, distance_matrix, n=2
        ),
        "add_route_removed": lambda sol: add_route_with_removed_bins_random(
            sol, candidate_removed_bins, stocks, vehicle_capacity
        ),
        "add_route_removed_consec": lambda sol: add_route_with_removed_bins_consecutive(
            sol, candidate_removed_bins, stocks, vehicle_capacity
        ),
        "relocate": lambda sol: (
            sol.__setitem__(idx := random.choice(range(len(sol))), relocate_within_route(sol[idx])) or sol
        ),
        "cross": lambda sol: cross_exchange(sol),
        "or-opt": lambda sol: (sol.__setitem__(idx := random.choice(range(len(sol))), or_opt_move(sol[idx])) or sol),
        "insert": lambda sol: _handle_insert(
            sol, data, stocks, vehicle_capacity, id_to_index, distance_matrix, candidate_removed_bins
        ),
    }

    handler = handlers.get(op)
    if handler:
        return handler(new_solution)
    return new_solution
