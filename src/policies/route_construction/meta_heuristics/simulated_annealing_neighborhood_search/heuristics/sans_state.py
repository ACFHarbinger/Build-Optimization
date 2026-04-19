"""
State and Cost utilities for Simulated Annealing.
"""

from logic.src.constants import (
    MAX_CAPACITY_PERCENT,
    PENALTY_MANDATORY_NODES_MISSED,
)
from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.distance import (
    compute_total_cost,
)


def compute_profit(
    solution, distance_matrix, id_to_index, data, vehicle_capacity, R, V, density, mandatory_bins, stocks=None
):
    """
    Calculate the profit of a solution.
    Profit = Revenue - Cost - Penalty
    """
    if stocks is None:
        stocks = dict(zip(data["#bin"], data["Stock"], strict=False))

    current_cost = compute_total_cost(solution, distance_matrix, id_to_index)

    # Calculate STRICT SIMULATOR PROFIT (Route 0 Only + Capacity Cutoff)
    real_kg = 0
    collected_mandatory = set()
    if len(solution) > 0 and len(solution[0]) > 2:
        route0 = solution[0]
        current_load = 0
        for b in route0:
            if b == 0:
                continue
            bin_kg = stocks.get(b, 0) * V * density / MAX_CAPACITY_PERCENT

            if current_load + bin_kg <= vehicle_capacity:
                current_load += bin_kg
                real_kg += bin_kg
                if mandatory_bins and b in mandatory_bins:
                    collected_mandatory.add(b)
            else:
                break

    current_revenue = real_kg * R

    # Mandatory Penalty: Force all mandatory_bins into the valid trunk of Route 0
    missed_mandatory = len(mandatory_bins) - len(collected_mandatory) if mandatory_bins else 0
    penalty_mandatory = missed_mandatory * PENALTY_MANDATORY_NODES_MISSED

    current_profit = current_revenue - current_cost - penalty_mandatory

    return current_profit, current_cost, current_revenue, real_kg
