"""
HGS Dispatcher module.

Dispatches to different HGS implementations based on configuration.
"""

from .hgs import HGSSolver
from .params import HGSParams
from .pyvrp_wrapper import solve_pyvrp


def run_hgs(dist_matrix, wastes, capacity, R, C, values, mandatory_nodes=None, x_coords=None, y_coords=None, *args):
    """
    Main HGS entry point with dispatching logic.

    Args:
        dist_matrix: Distance matrix.
        wastes: Bin wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        values: Dictionary of parameters and config.
        mandatory_nodes: List of local node indices that MUST be visited.
        *args: Additional arguments (ignored or passed through).

    Returns:
        Tuple[List[List[int]], float, float]: Best routes, profit, and cost.
    """
    # Create HGSParams using Vidal 2022 parameter names
    params = HGSParams(
        # Core HGS parameters (Vidal 2022)
        time_limit=values.get("time_limit", 0.0),
        mu=values.get("mu", 25),
        n_offspring=values.get("n_offspring", values.get("lambda_param", 40)),
        nb_elite=values.get("nb_elite", 4),
        nb_close=values.get("nb_close", 5),
        nb_granular=values.get("nb_granular", 20),
        target_feasible=values.get("target_feasible", 0.2),
        n_iterations_no_improvement=values.get("n_iterations_no_improvement", 20000),
        # Genetic operators
        mutation_rate=values.get("mutation_rate", 1.0),
        repair_probability=values.get("repair_probability", 0.5),
        crossover_rate=values.get("crossover_rate", 1.0),
        # Local search
        local_search_iterations=values.get("local_search_iterations", 500),
        max_vehicles=values.get("max_vehicles", 0),
        # Penalty management
        initial_penalty_capacity=values.get("initial_penalty_capacity", 1.0),
        penalty_increase=values.get("penalty_increase", 1.2),
        penalty_decrease=values.get("penalty_decrease", 0.85),
        vrpp=values.get("vrpp", True),
        profit_aware_operators=values.get("profit_aware_operators", False),
        use_cross_exchange=values.get("use_cross_exchange", False),
        use_lambda_interchange=values.get("use_lambda_interchange", False),
        lambda_max=values.get("lambda_max", 0),
        use_ejection_chains=values.get("use_ejection_chains", False),
        seed=values.get("seed", 42),
        engine=values.get("engine", "custom"),
        restart_timer=values.get("restart_timer", 0.0),
    )
    if params.engine == "pyvrp":
        return solve_pyvrp(dist_matrix, wastes, capacity, R, C, values)

    if len(dist_matrix) <= 1:
        return [], 0.0, 0.0

    if len(dist_matrix) == 2:
        d = wastes.get(1, 0)
        if d <= capacity:
            # Fix 5: Calculate simple profit = revenue - cost
            cost = dist_matrix[0][1] + dist_matrix[1][0]
            revenue = d * R
            total_cost = C * cost
            profit = revenue - total_cost
            return [[1]], profit, total_cost
        else:
            return [], 0.0, 0.0

    solver = HGSSolver(dist_matrix, wastes, capacity, R, C, params, mandatory_nodes, x_coords, y_coords)
    return solver.solve()
