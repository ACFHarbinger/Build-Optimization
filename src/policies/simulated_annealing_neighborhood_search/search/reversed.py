"""
Reversed Search Module.

This module implements a search strategy that explores the solution space
by reversing routes or segments of routes.

Attributes:
    None

Example:
    >>> from logic.src.policies.simulated_annealing_neighborhood_search.search.reversed import ReversedSearch
    >>> search = ReversedSearch(...)
    >>> result = search.run()
"""

from copy import deepcopy

from logic.src.policies.simulated_annealing_neighborhood_search.common.objectives import (
    compute_profit,
    compute_real_profit,
)
from logic.src.policies.simulated_annealing_neighborhood_search.search.deterministic import (
    _evaluate_move,
)


def local_search_reversed(
    previous_solution,
    p_vehicle,
    p_load,
    p_route_difference,
    p_shift,
    data,
    distance_matrix,
    values,
):
    """
    Similar to local_search_2 but applies operators in a different order or variant.

    Args:
        previous_solution (List[List[int]]): Solution to improve.
        p_vehicle (float): Vehicle penalty.
        p_load (float): Load penalty.
        p_route_difference (float): Imbalance penalty.
        p_shift (float): Shift penalty.
        data (pd.DataFrame): Bin weights data.
        distance_matrix (np.ndarray): Shortest path matrix.
        values (Dict): Parameter dictionary.

    Returns:
        List[List[int]]: Modified routing solution.
    """
    previous_profit = compute_profit(
        previous_solution,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )
    routes_list = deepcopy(previous_solution)

    # Loop over routes in reversed order
    for idx_route in reversed(range(len(previous_solution))):
        current_route_data = previous_solution[idx_route]

        # Loop bins in reversed order (excluding last depot)
        bins_to_check = current_route_data[:-1]

        for val in reversed(bins_to_check):
            position = current_route_data.index(val)
            row = list(distance_matrix[val][:])
            row_new = sorted(row)

            candidates = []
            count = 0
            for dist in row_new[1:]:
                b_idx = row.index(dist)
                if b_idx != 0:
                    candidates.append(b_idx)
                    count += 1
                if count >= 50:
                    break

            for bin_to_move in candidates:
                new_profit, improved = _evaluate_move(
                    routes_list,
                    idx_route,
                    position,
                    bin_to_move,
                    previous_solution,
                    previous_profit,
                    p_vehicle,
                    p_load,
                    p_route_difference,
                    p_shift,
                    data,
                    distance_matrix,
                    values,
                )

                if improved:
                    previous_solution = deepcopy(routes_list)
                    previous_profit = new_profit
                else:
                    routes_list = deepcopy(previous_solution)

    solution_after_local_search_reversed = deepcopy(previous_solution)
    profit_after_local_search_reversed = compute_profit(
        solution_after_local_search_reversed,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )
    LS_reversed_profit = compute_real_profit(
        solution_after_local_search_reversed, p_vehicle, data, distance_matrix, values
    )
    return (
        solution_after_local_search_reversed,
        profit_after_local_search_reversed,
        LS_reversed_profit,
    )
