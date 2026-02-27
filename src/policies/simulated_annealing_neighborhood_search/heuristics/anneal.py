"""
Simulated Annealing loop for the Look-Ahead policy.
"""

import math
import time
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from logic.src.policies.simulated_annealing_neighborhood_search.common.check import (
    check_bins_overflowing_feasibility,
    check_solution_admissibility,
)
from logic.src.policies.simulated_annealing_neighborhood_search.common.objectives import compute_profit
from logic.src.policies.simulated_annealing_neighborhood_search.search.random_search import local_search


def run_annealing_loop(
    initial_solution: List[List[int]],
    data: pd.DataFrame,
    distance_matrix: np.ndarray,
    must_go_bins: List[int],
    values: Dict,
    n_bins: int,
    chosen_combination: Tuple,
    time_limit: float,
) -> Tuple[List[List[int]], List[int]]:
    """
    Execute the simulated annealing loop to find an improved solution.

    Args:
        initial_solution: Seed solution.
        data: Bin data.
        distance_matrix: Distances.
        must_go_bins: Mandatory nodes.
        values: parameters.
        n_bins: Problem size.
        chosen_combination: SA parameters.
        time_limit: Max runtime.

    Returns:
        Tuple of (optimized_solution, removed_bins).
    """
    number_iterations = chosen_combination[0]
    T_initial = chosen_combination[1]
    T_param = chosen_combination[2]
    p_vehicle = chosen_combination[3]
    p_load = chosen_combination[4]
    p_route_difference = chosen_combination[5]
    p_shift = chosen_combination[6]

    removed_bins: List[int] = []
    previous_sol = initial_solution
    previous_sol_profit = compute_profit(
        previous_sol,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    routes_list = deepcopy(initial_solution)
    tic = time.perf_counter()

    for i in range(1, number_iterations + 1):
        (
            chosen_procedure,
            _,
            bin_to_remove,
            bin_to_add,
            bins_to_remove_random,
            bins_to_remove_consecutive,
            bins_to_add_random,
            bins_to_add_consecutive,
            bins_random,
            bins_consecutive,
        ) = local_search(routes_list, removed_bins, distance_matrix, must_go_bins)

        # Compute profit
        current_sol_profit = compute_profit(
            routes_list,
            p_vehicle,
            p_load,
            p_route_difference,
            p_shift,
            data,
            distance_matrix,
            values,
        )

        status = check_bins_overflowing_feasibility(
            data,
            routes_list,
            n_bins,
            values["perc_bins_can_overflow"],
            values["E"],
            values["B"],
        )
        _ = check_solution_admissibility(routes_list, removed_bins, n_bins)

        if status == "pass":
            delta = current_sol_profit - previous_sol_profit
            if delta >= 0:
                previous_sol = deepcopy(routes_list)
                previous_sol_profit = current_sol_profit

                # Update removed bins set based on accepted move
                if delta == 0:
                    _update_removed_bins(
                        chosen_procedure,
                        removed_bins,
                        bin_to_remove,
                        bin_to_add,
                        bins_to_remove_random,
                        bins_to_remove_consecutive,
                        bins_to_add_random,
                        bins_to_add_consecutive,
                        bins_random,
                        bins_consecutive,
                    )
            else:
                T = T_initial / (i**T_param)
                if math.exp(delta / T) >= np.random.uniform(0, 1):
                    previous_sol = deepcopy(routes_list)
                    previous_sol_profit = current_sol_profit
                else:
                    _rollback_removed_bins(
                        chosen_procedure,
                        removed_bins,
                        bin_to_remove,
                        bin_to_add,
                        bins_to_remove_random,
                        bins_to_remove_consecutive,
                        bins_to_add_random,
                        bins_to_add_consecutive,
                        bins_random,
                        bins_consecutive,
                    )
        else:
            _rollback_removed_bins(
                chosen_procedure,
                removed_bins,
                bin_to_remove,
                bin_to_add,
                bins_to_remove_random,
                bins_to_remove_consecutive,
                bins_to_add_random,
                bins_to_add_consecutive,
                bins_random,
                bins_consecutive,
            )

        routes_list = deepcopy(previous_sol)
        if (time.perf_counter() - tic) > time_limit:
            break

    return previous_sol, removed_bins


def _update_removed_bins(proc, removed, b_rem, b_add, bs_rem_rnd, bs_rem_con, bs_add_rnd, bs_add_con, bs_rnd, bs_con):
    """Internal helper to update removed bins set on move acceptance."""
    if proc == "Drop bin" and b_rem is not None:
        removed.remove(b_rem)
    elif proc == "Add bin" and b_add is not None:
        removed.append(b_add)
    elif proc == "Remove n bins random" and bs_rem_rnd:
        for n in bs_rem_rnd:
            removed.remove(n)
    elif proc == "Remove n bins consecutive" and bs_rem_con:
        for o in bs_rem_con:
            removed.remove(o)
    elif proc == "Add n bins random" and bs_add_rnd:
        for r in bs_add_rnd:
            removed.append(r)
    elif proc == "Add n bins consecutive" and bs_add_con:
        for s in bs_add_con:
            removed.append(s)
    elif proc == "Add route with removed bins random":
        for t in bs_rnd:
            removed.append(t)
    elif proc == "Add route with removed bins consecutive":
        for u in bs_con:
            removed.append(u)


def _rollback_removed_bins(proc, removed, b_rem, b_add, bs_rem_rnd, bs_rem_con, bs_add_rnd, bs_add_con, bs_rnd, bs_con):
    """Internal helper to rollback removed bins set on move rejection."""
    # Note: In the original code, some logic was slightly inconsistent or redundant.
    # We follow the original route_search.py logic closely.
    if proc == "Drop bin" and b_rem is not None:
        removed.remove(b_rem)
    elif proc == "Add bin" and b_add is not None:
        removed.append(b_add)
    elif proc == "Remove n bins random" and bs_rem_rnd:
        for n in bs_rem_rnd:
            removed.remove(n)
    elif proc == "Remove n bins consecutive" and bs_rem_con:
        for o in bs_rem_con:
            removed.remove(o)
    elif proc == "Add n bins random" and bs_add_rnd:
        for r in bs_add_rnd:
            removed.append(r)
    elif proc == "Add n bins consecutive" and bs_add_con:
        for s in bs_add_con:
            removed.append(s)
    elif proc == "Add route with removed bins random":
        for t in bs_rnd:
            removed.append(t)
    elif proc == "Add route with removed bins consecutive":
        for u in bs_con:
            removed.append(u)
