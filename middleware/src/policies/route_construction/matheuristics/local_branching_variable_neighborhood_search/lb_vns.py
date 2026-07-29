"""
Local Branching with Variable Neighborhood Search (LB-VNS) matheuristic solver.

This implementation follows the hybrid approach proposed by Hansen et al. (2006),
which integrates the metaheuristic structure of Variable Neighborhood Search (VNS)
with the local search capabilities of Local Branching (LB).

Algorithm Overview:
    1. Initialization: Find an initial feasible solution (incumbent).
    2. VNS Cycle:
        a. Shaking: Randomly perturb the incumbent to find a starting point
           in a neighborhood N_k (defined by Hamming distance k).
        b. Local Search: Apply Local Branching from the shaken solution
           to find a local optimum or improved solution.
        c. Move/Stay: If improvement is found, update incumbent and reset k.
           Otherwise, increase k (neighborhood expansion).

Reference:
    Hansen, P., Mladenović, N., & Urošević, D. (2006).
    "Variable neighborhood search and local branching".
    Computers & Operations Research.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    gp: Any = None  # type: ignore[assignment,no-redef]
    GRB: Any = None  # type: ignore[misc,assignment,no-redef]

from logic.src.policies.route_construction.matheuristics.kernel_search.solver import (
    _dfj_subtour_elimination_callback,
    _reconstruct_tour,
    _set_mip_start,
    _setup_ks_model,
)
from logic.src.policies.route_construction.matheuristics.local_branching.lb import (
    _add_local_branching_constraint,
)
from logic.src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params import (
    LBVNSParams,
)


def _shake_solution_gurobi(
    model: gp.Model,
    x: Dict[Tuple[int, int], gp.Var],
    y: Dict[int, gp.Var],
    incumbent_x: Dict[Tuple[int, int], float],
    incumbent_y: Dict[int, float],
    k: int,
    seed: int = 42,
) -> Tuple[Optional[Dict[Tuple[int, int], float]], Optional[Dict[int, float]]]:
    """
    Perform the Shaking phase by finding a solution exactly at Hamming distance k.

    Following Hanafi et al. (2010) and Hansen et al. (2006), the shaking step
    aims to find a feasible solution s' such that delta(s, s') = k.
    """
    # 1. Save original objective
    orig_obj = model.getObjective()

    # 2. Define Hamming distance (delta)
    lhs = gp.LinExpr()
    # sum_{j: incumbent=0} x_j + sum_{j: incumbent=1} (1 - x_j)
    for key, var in x.items():
        if incumbent_x.get(key, 0.0) < 0.5:
            lhs.addTerms(1.0, var)
        else:
            lhs.addConstant(1.0)
            lhs.addTerms(-1.0, var)
    for key, var in y.items():  # type: ignore[assignment]
        if incumbent_y.get(key, 0.0) < 0.5:  # type: ignore[call-overload]
            lhs.addTerms(1.0, var)
        else:
            lhs.addConstant(1.0)
            lhs.addTerms(-1.0, var)

    # Shaking logic: exactly k or in [k-1, k+1] if k is large
    shake_cut = model.addConstr(lhs == k, name="shake_cut_exact")

    # 3. Random objective to find a feasible point on the boundary
    rng = np.random.default_rng(seed)
    random_obj = gp.LinExpr()
    for var in model.getVars():
        if var.VType in [GRB.BINARY, GRB.INTEGER]:
            random_obj.addTerms(rng.standard_normal(), var)

    model.setObjective(random_obj, GRB.MINIMIZE)
    model.setParam("SolutionLimit", 1)
    model.setParam("TimeLimit", 10.0)

    model.optimize(_dfj_subtour_elimination_callback)

    res_x, res_y = None, None
    if model.SolCount > 0:
        res_x = {key: var.X for key, var in x.items()}
        res_y = {key: var.X for key, var in y.items()}
    else:
        # Fallback: try lhs <= k if exact is too hard
        model.remove(shake_cut)
        shake_cut = model.addConstr(lhs <= k, name="shake_cut_le")
        model.optimize(_dfj_subtour_elimination_callback)
        if model.SolCount > 0:
            res_x = {key: var.X for key, var in x.items()}
            res_y = {key: var.X for key, var in y.items()}

    # 4. Cleanup
    model.remove(shake_cut)
    model.setObjective(orig_obj)
    model.setParam("SolutionLimit", 2000000000)  # Reset to default large value

    return res_x, res_y


def run_lb_vns_gurobi(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
    mip_gap: float = 0.01,
    seed: int = 42,
    env: Optional[gp.Env] = None,
    params: Optional[LBVNSParams] = None,
) -> Tuple[List[int], float, float]:
    """
    Solve the Vehicle Routing Problem with Profits (VRPP) using LB-VNS.

    LB-VNS iterates through neighborhoods of increasing size (Hamming distance).
    In each iteration, it 'shakes' the incumbent to a new point and then
    intensifies the search using Local Branching.

    Mathematical Formulation:
        The underlying model is a Prize-Collecting VRP / VRP with Profits,
        implemented using MTZ (Miller-Tucker-Zemlin) subtour elimination constraints.

    Args:
        dist_matrix (np.ndarray): Symmetric distance matrix between all nodes.
        wastes (Dict[int, float]): Dictionary mapping node indices to profits.
        capacity (float): Maximum vehicle load.
        R (float): Revenue multiplier for node collection.
        C (float): Cost multiplier for distance traveled.
        mandatory_nodes (List[int]): Nodes that MUST be included in the tour.
        k_min (int): Minimum Hamming distance for early search stages.
        k_max (int): Maximum Hamming distance (exploration ceiling).
        k_step (int): Increment for 'k' when no improvement is found.
        time_limit (float): Overall wall-clock budget for the search.
        time_limit_per_lb (float): Time budget for each Local Branching call.
        max_lb_iterations (int): Loop limit for internal Local Branching.
        mip_gap (float): Optimality gap for sub-problem termination.
        seed (int): Global seed for randomization.
        env (Optional[gp.Env]): Shared Gurobi environment to reduce startup overhead.

    Returns:
        Tuple[List[int], float, float]:
            - tour: The node sequence representing the best found tour.
            - objective_value: The final mathematical objective score.
            - total_distance: The physical distance cost of the tour.
    """
    # Initialize the Gurobi model
    model = gp.Model("LB_VNS_VRPP", env=env) if env else gp.Model("LB_VRPP")
    model.setParam("OutputFlag", 0)  # Silent mode
    # (blank line)
    # Initialize params if not provided
    params = params or LBVNSParams()
    # (blank line)
    model.setParam("Seed", params.seed)
    model.setParam("MIPGap", params.mip_gap)

    # Setup modular acceptance criterion (fallback if params not provided)
    if params is not None and params.acceptance_criterion is not None:
        acceptance_criterion = params.acceptance_criterion
    else:
        # Avoid circular import if needed, or use a default
        from logic.src.policies.route_construction.acceptance_criteria.only_improving import OnlyImproving

        acceptance_criterion = OnlyImproving()

    # 1. Setup mathematical formulation with DFJ lazy constraints (no MTZ)
    # This creates the variables x (edges) and y (nodes) and the base constraints.
    # CRITICAL: use_binary_vars=True because LB-VNS needs integer solutions, not LP relaxation
    x, y = _setup_ks_model(model, dist_matrix, wastes, capacity, R, C, mandatory_nodes, use_binary_vars=True)

    # 1.5. Warm-start with greedy heuristic
    # This is CRITICAL for LB-VNS: we need an incumbent to define the k-neighborhood
    _set_mip_start(model, x, y, dist_matrix, wastes, capacity, R, C, mandatory_nodes)

    # 2. Find an initial feasible solution
    # A short initial B&B run to establish a baseline (incumbent).
    # We allocate up to 30% of the budget (or at least 20s if budget allows) for this critical step
    initial_alloc = min(params.time_limit * 0.8, max(20.0, params.time_limit * 0.3))
    model.setParam("TimeLimit", initial_alloc)
    model.optimize(_dfj_subtour_elimination_callback)

    if model.SolCount == 0:
        # If no solution found, try a slightly longer emergency solve.
        fallback_alloc = min(params.time_limit * 0.8, max(15.0, params.time_limit * 0.5))
        model.setParam("TimeLimit", fallback_alloc)
        model.optimize(_dfj_subtour_elimination_callback)
        if model.SolCount == 0:
            return [0, 0], 0.0, 0.0

    current_best_obj = model.ObjVal
    incumbent_x = {key: var.X for key, var in x.items()}
    incumbent_y = {key: var.X for key, var in y.items()}

    # Setup modular acceptance criterion
    acceptance_criterion.setup(current_best_obj)

    # 3. MAIN VARIABLE NEIGHBORHOOD SEARCH LOOP
    # =========================================================================
    start_time = time.process_time()
    k = params.k_min

    while k <= params.k_max:
        elapsed = time.process_time() - start_time
        if elapsed > params.time_limit:
            # Terminate if the global budget is exhausted.
            break

        # --- PHASE 1: SHAKING (Exploration) ---
        # Find a diversified starting point in the k-neighborhood.
        shaken_x, shaken_y = _shake_solution_gurobi(model, x, y, incumbent_x, incumbent_y, k, seed + k)  # type: ignore[bad-argument-type]

        if shaken_x is None:
            # If the neighborhood is infeasible (e.g., k is too small to flip mandatory edges),
            # expand the neighborhood and try again.
            k += params.k_step
            continue

        # --- PHASE 2: LOCAL SEARCH (Intensification) ---
        # Apply Local Branching to improve the shaken solution.
        # k_ls is a smaller, fixed radius (10) for focused intensification.
        k_ls = 10

        # Set current solution as starting point for LB iterations
        current_ls_x = shaken_x
        current_ls_y = shaken_y
        current_ls_obj = current_best_obj

        # Internal Local Branching loop for intensification
        lb_iter = 0
        while lb_iter < params.max_lb_iterations:
            remaining_time = params.time_limit - (time.process_time() - start_time)
            if remaining_time <= 0:
                break

            # Add Local Branching constraint centered on current_ls solution
            ls_cut = _add_local_branching_constraint(model, x, y, current_ls_x, current_ls_y, k_ls)  # type: ignore[arg-type]

            # Warm-start the solver with the current solution
            for key, var in x.items():
                var.Start = current_ls_x.get(key, 0.0)
            for key, var in y.items():  # type: ignore[assignment]
                var.Start = current_ls_y.get(key, 0.0)  # type: ignore[call-overload,union-attr]

            iter_time = min(params.time_limit_per_lb / params.max_lb_iterations, remaining_time)
            model.setParam("TimeLimit", iter_time)
            model.optimize(_dfj_subtour_elimination_callback)

            # Check if we found a better solution in this LB iteration
            if model.SolCount > 0 and model.ObjVal > current_ls_obj + 1e-4:
                # Update local search incumbent
                current_ls_obj = model.ObjVal
                current_ls_x = {key: var.X for key, var in x.items()}
                current_ls_y = {key: var.X for key, var in y.items()}
                # Continue intensifying from this improved solution
            else:
                # No improvement in this neighborhood, exit LB loop
                model.remove(ls_cut)
                break

            model.remove(ls_cut)
            lb_iter += 1

        # --- PHASE 3: NEIGHBORHOOD CHANGE (Move Acceptance) ---
        # Delegate decision to injected criterion
        is_accepted, _ = acceptance_criterion.accept(
            current_obj=current_best_obj,
            candidate_obj=current_ls_obj,
            iteration=int((time.process_time() - start_time) / params.time_limit * 100),  # Progress-based iteration
            max_iterations=100,
        )

        if is_accepted:
            # Moving: Update and intensification restarts.
            current_best_obj = current_ls_obj
            incumbent_x = current_ls_x
            incumbent_y = current_ls_y  # type: ignore[assignment]
            k = params.k_min  # Reset VNS to smallest neighborhood
        else:
            # Staying: Neighborhood N_k exhausted. Expand to diversify further.
            k += params.k_step

        # Step criterion
        acceptance_criterion.step(
            current_obj=current_best_obj,
            candidate_obj=current_ls_obj,
            accepted=is_accepted,
        )

        # Cleanup warm-start values
        for var in model.getVars():
            var.Start = GRB.UNDEFINED

    # 4. FINAL SOLUTION EXTRACTION
    # =========================================================================
    # After the loop terminates, reconstruct the best found tour sequence.
    tour, cost = _reconstruct_tour(len(dist_matrix), incumbent_x, dist_matrix)

    return tour, float(current_best_obj), float(cost)
