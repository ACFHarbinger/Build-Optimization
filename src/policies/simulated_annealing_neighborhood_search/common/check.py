"""
Consistency and feasibility validation for look-ahead routing solutions.

This module provides routines to verify that routing solutions adhere to
physical constraints (tanker capacity) and temporal predictions (overflow
risk). It is used during the local search process to prune inadmissible
mutations and ensure valid final outputs for the simulator.
"""

# Functions for condition checks


# Check route feasibility
# function to check routes feasibility regarding number of overflowing bins allowed
# the visited number of bins to which stock + accumulation_rate >= capacity has to be greater
# or equal than the number of bins that verify this condition minus the total number of bins
# multiplied by the percentage of bins that can overflow
def check_bins_overflowing_feasibility(data, routes_list, number_of_bins, perc_bins_can_overflow, E, B):
    """
    Check if the solution is feasible regarding the number of overflowing bins.

    A solution is considered feasible if the number of bins currently in the
    routes that were predicted to overflow (or are near capacity) satisfies
     the tolerance threshold.

    Args:
        data (pd.DataFrame): Bin data containing 'Stock' and 'Accum_Rate'.
        routes_list (List[List[int]]): Current routing solution.
        number_of_bins (int): Total number of bins in the area.
        perc_bins_can_overflow (float): Threshold (delta) for allowed overflow.
        E (float): Bin volume.
        B (float): Bin density.

    Returns:
        str: 'pass' if feasible, 'fail' otherwise.
    """
    bins_overflowing = []
    for _index, row in data.iterrows():
        if row["Stock"] + row["Accum_Rate"] >= E * B:
            bins_overflowing.append(row["#bin"])

    number_bins_overflowing = len(bins_overflowing)
    check = number_bins_overflowing - number_of_bins * perc_bins_can_overflow
    overflowing_bins_in_routes = []
    for i in bins_overflowing:
        if any(i in a for a in routes_list):
            overflowing_bins_in_routes.append(1)
        else:
            overflowing_bins_in_routes.append(0)

    total_ovf_bins_in_routes = sum(overflowing_bins_in_routes)
    status = "pass" if total_ovf_bins_in_routes >= check else "fail"
    return status


# Check route admissibility
# checks if every function is working fine and admissible solutions are being generated
def check_solution_admissibility(routes_list, removed_bins, number_of_bins):
    """
    Verify if all bins are accounted for in the routes or removed set.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        removed_bins (List[int]): Set of bins not collected.
        number_of_bins (int): Total number of collectable bins.

    Returns:
        bool: True if the solution is consistent, False otherwise.
    """
    total_length = 0
    for i in routes_list:
        total_length += len(i)
    admissible = total_length + len(removed_bins) == number_of_bins + 2 * len(routes_list)
    return admissible
