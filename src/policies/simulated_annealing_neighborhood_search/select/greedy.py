"""
Greedy selection strategies for bin management.
"""

from copy import deepcopy

from logic.src.policies.simulated_annealing_neighborhood_search.common.objectives import (
    compute_profit,
    compute_real_profit,
)


def insert_bins(
    routes_list,
    removed_bins,
    p_vehicle,
    p_load,
    p_route_difference,
    p_shift,
    data,
    distance_matrix,
    values,
):
    """
    Greedily insert bins from the removed set into the current routes if it improves profit.

    Continues until no more profitable insertions are found.

    Args:
        routes_list (List[List[int]]): Initial routes.
        removed_bins (List[int]): Available bins.
        p_vehicle (float): Vehicle penalty.
        p_load (float): Load penalty.
        p_route_difference (float): Difference penalty.
        p_shift (float): Shift penalty.
        data (pd.DataFrame): Weights context.
        distance_matrix (np.ndarray): Distance data.
        values (Dict): Constants.

    Returns:
        List[List[int]]: Optimized routing solution.
    """
    inserted_bins = []
    previous_solution = deepcopy(routes_list)
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
    for _it, a in enumerate(removed_bins):
        for it1, i in enumerate(previous_solution):
            i = routes_list[it1]
            for it2, _j in enumerate(i[0 : len(i) - 1]):
                i = routes_list[it1]
                for b in routes_list:
                    if a in b:
                        b.remove(a)
                position = it2 + 1
                i.insert(position, a)
                new_profit = compute_profit(
                    routes_list,
                    p_vehicle,
                    p_load,
                    p_route_difference,
                    p_shift,
                    data,
                    distance_matrix,
                    values,
                )
                if new_profit > previous_profit:
                    previous_solution = deepcopy(routes_list)
                    previous_profit = new_profit
                else:
                    routes_list = deepcopy(previous_solution)
    for c in removed_bins:
        for m in previous_solution:
            if c in m:
                inserted_bins.append(c)

    for ins in inserted_bins:
        removed_bins.remove(ins)

    previous_real_profit = compute_real_profit(previous_solution, p_vehicle, data, distance_matrix, values)
    return previous_solution, previous_profit, previous_real_profit, removed_bins


def remove_bins_end(
    routes_list,
    removed_bins,
    p_vehicle,
    p_load,
    p_route_difference,
    p_shift,
    data,
    bins_cannot_removed,
    distance_matrix,
    values,
):
    """
    Greedily remove bins from the current routes if it increases total profit.

    Args:
        routes_list (List[List[int]]): Initial routes.
        removed_bins (List[int]): Target set for dropped bins.
        p_vehicle (float): Vehicle penalty.
        p_load (float): Load penalty.
        p_route_difference (float): Difference penalty.
        p_shift (float): Shift penalty.
        data (pd.DataFrame): Bin metrics.
        bins_cannot_removed (List[int]): Mandatory bins.
        distance_matrix (np.ndarray): Metric matrix.
        values (Dict): Constants.

    Returns:
        List[List[int]]: Optimized routing solution.
    """
    previous_solution = deepcopy(routes_list)
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
    for it, i in enumerate(previous_solution):
        i = routes_list[it]
        for j in i[1 : len(i) - 1]:
            i = routes_list[it]
            if j not in bins_cannot_removed:
                i.remove(j)
            new_profit = compute_profit(
                routes_list,
                p_vehicle,
                p_load,
                p_route_difference,
                p_shift,
                data,
                distance_matrix,
                values,
            )
            if new_profit > previous_profit:
                previous_solution = deepcopy(routes_list)
                previous_profit = new_profit
                removed_bins.append(j)
            else:
                routes_list = deepcopy(previous_solution)
    previous_real_profit = compute_real_profit(previous_solution, p_vehicle, data, distance_matrix, values)
    return previous_solution, previous_profit, previous_real_profit, removed_bins
