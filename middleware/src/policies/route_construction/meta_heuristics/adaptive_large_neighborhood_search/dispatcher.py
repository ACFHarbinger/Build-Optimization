"""
ALNS Dispatcher module.

Dispatches to different ALNS implementations based on configuration.
"""

from .alns import ALNSSolver
from .alns_package import run_alns_package
from .ortools_wrapper import run_alns_ortools
from .params import ALNSParams


def run_alns(dist_matrix, wastes, capacity, R, C, values, mandatory_nodes=None, recorder=None, *args):
    """
    Main ALNS entry point with dispatching to different algorithm variants.

    Args:
        dist_matrix: Distance matrix.
        wastes: Bin wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        values: Dictionary of parameters and config.
        mandatory_nodes: List of mandatory node indices.
        recorder: Optional telemetry recorder.
        *args: Additional arguments (ignored or passed through).

    Returns:
        Tuple[List[List[int]], float, float]: Best routes, profit, and cost.
    """
    variant = values.get("variant") or values.get("engine") or "custom"

    if variant == "package":
        return run_alns_package(dist_matrix, wastes, capacity, R, C, values)
    elif variant == "ortools":
        return run_alns_ortools(dist_matrix, wastes, capacity, R, C, values)

    # Default: Custom internal ALNS solver
    params = ALNSParams(
        time_limit=values.get("time_limit", 10),
        max_iterations=values.get("max_iterations", 2000),
        start_temp=values.get("start_temp", 100.0),
        cooling_rate=values.get("cooling_rate", 0.995),
        reaction_factor=values.get("reaction_factor", 0.1),
        min_removal=values.get("min_removal", 4),
        max_removal_pct=values.get("max_removal_pct", 0.3),
        segment_size=values.get("segment_size", 100),
        noise_factor=values.get("noise_factor", 0.025),
        worst_removal_randomness=values.get("worst_removal_randomness", 3.0),
        shaw_randomization=values.get("shaw_randomization", 6.0),
        max_removal_cap=values.get("max_removal_cap", 100),
        regret_pool=values.get("regret_pool", "regret234"),
        sigma_1=values.get("sigma_1", 33.0),
        sigma_2=values.get("sigma_2", 9.0),
        sigma_3=values.get("sigma_3", 13.0),
        vrpp=values.get("vrpp", True),
        profit_aware_operators=values.get("profit_aware_operators", False),
        seed=values.get("seed", 42),
    )
    solver = ALNSSolver(dist_matrix, wastes, capacity, R, C, params, mandatory_nodes, recorder=recorder)
    return solver.solve()
