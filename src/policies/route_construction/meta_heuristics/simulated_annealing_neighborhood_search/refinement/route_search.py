"""
Routing search strategies and orchestration.
Contains the main Simulated Annealing and Local Search loops for the LookAhead policy.
"""

from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.routes import (
    uncross_arcs_in_routes,
)
from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.solution_initialization import (
    find_initial_solution,
)
from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.anneal import (
    run_annealing_loop,
)
from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.refinement.refinement import (
    refine_solution,
)

from .rebalancing import rebalance_solution


def find_solutions(
    data,
    bins_coordinates,
    distance_matrix,
    chosen_combination,
    mandatory_bins,
    values,
    n_bins,
    points,
    time_limit,
    rng,
    np_rng,
):
    """
    Find high-quality routing solutions using a randomized local search procedure.

    Args:
        data (pd.DataFrame): Bin and weight context.
        bins_coordinates (List): Locations.
        distance_matrix (np.ndarray): distances.
        chosen_combination (Tuple): Parameter set (vehicle penalty, load penalty, etc).
        mandatory_bins (List[int]): Nodes that must be visited.
        values (Dict): Global constants.
        n_bins (int): Problem size.
        points (Dict): Coordinates map.
        time_limit (float): Max execution time.
        rng (random.Random): Random number generator.
        np_rng (np.random.Generator): Numpy random number generator.

    Returns:
        List[List[int]]: Optimized routing solution.
    """
    p_iterations = chosen_combination[0]
    p_vehicle = chosen_combination[3]
    p_load = chosen_combination[4]
    p_route_difference = chosen_combination[5]
    p_shift = chosen_combination[6]

    # 1. Initial Solution
    initial_solution = find_initial_solution(
        data,
        bins_coordinates,
        distance_matrix,
        n_bins,
        values["vehicle_capacity"],
        values["E"],
        values["B"],
        rng,
    )
    initial_solution, _, _ = uncross_arcs_in_routes(
        initial_solution,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        points,
        distance_matrix,
        values,
    )

    # 2. Simulated Annealing Phase
    sa_sol, removed_bins = run_annealing_loop(
        initial_solution,
        data,
        distance_matrix,
        mandatory_bins,
        values,
        n_bins,
        chosen_combination,
        time_limit,
        rng,
        np_rng,
    )

    # 3. Refinement Phase (LS/Uncross iterative loops)
    refined_sol, _ = refine_solution(
        sa_sol,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        points,
        distance_matrix,
        values,
        iterations=p_iterations,
    )

    # 4. Rebalancing Phase (Remove/Insert iterative loops)
    final_routes, final_profit, final_removed_bins = rebalance_solution(
        refined_sol,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        mandatory_bins,
        distance_matrix,
        values,
        iterations=p_iterations,
    )

    return final_routes, final_profit, final_removed_bins
