"""
Solution rebalancing strategies for the look-ahead policy.
Iteratively removes and re-inserts bins to optimize route density and feasibility.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from logic.src.policies.simulated_annealing_neighborhood_search.select import insert_bins, remove_bins_end


def rebalance_solution(
    solution: List[List[int]],
    removed_bins: List[int],
    p_vehicle: float,
    p_load: float,
    p_route_difference: float,
    p_shift: float,
    data: pd.DataFrame,
    must_go_bins: List[int],
    distance_matrix: np.ndarray,
    values: Dict,
    iterations: int = 10,
) -> Tuple[List[List[int]], float, List[int]]:
    """
    Iteratively remove and re-insert bins to balance the solution.

    Args:
        solution: Current routing solution.
        removed_bins: List of bins currently excluded.
        p_vehicle: Vehicle penalty.
        p_load: Load penalty.
        p_route_difference: Imbalance penalty.
        p_shift: Shift penalty.
        data: Bin data.
        must_go_bins: Mandatory nodes.
        distance_matrix: Distances.
        values: parameters.
        iterations: Number of swap loops.

    Returns:
        Tuple of (balanced_solution, final_profit, final_removed_bins).
    """
    current_sol = solution
    current_removed = removed_bins
    final_profit = 0.0

    for _ in range(iterations):
        # 1. Remove Bins from ends/bad positions
        current_sol, _, _, current_removed = remove_bins_end(
            current_sol,
            current_removed,
            p_vehicle,
            p_load,
            p_route_difference,
            p_shift,
            data,
            must_go_bins,
            distance_matrix,
            values,
        )

        # 2. Re-insert potentially better combinations
        current_sol, _, final_profit, current_removed = insert_bins(
            current_sol,
            current_removed,
            p_vehicle,
            p_load,
            p_route_difference,
            p_shift,
            data,
            distance_matrix,
            values,
        )

    return current_sol, final_profit, current_removed
