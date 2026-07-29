"""
Adaptive Kernel Search (AKS) matheuristic solver for VRPP.

Reference:
    Guastaroba, G., Savelsbergh, M., & Speranza, M. G. (2017). "Adaptive Kernel
    Search: A heuristic for solving Mixed Integer linear Programs".
    European Journal of Operational Research.
"""

import random
from typing import Dict, List, Optional, Set, Tuple

import gurobipy as gp
import numpy as np

from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes
from logic.src.policies.route_construction.matheuristics.kernel_search.solver import (
    _dfj_subtour_elimination_callback,
    _reconstruct_tour,
    _root_node_callback,
    _set_mip_start,
    _setup_ks_model,
)
from logic.src.tracking.viz_mixin import PolicyStateRecorder


def _get_partitioned_vars_aks(
    model: gp.Model,
    x: Dict[Tuple[int, int], gp.Var],
    y: Dict[int, gp.Var],
    initial_kernel_size: int,
    bucket_size: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
) -> Tuple[List[gp.Var], List[gp.Var], List[List[gp.Var]]]:
    """
    Solve the MILP root node relaxation and partition variables into Kernel and Buckets.
    Ranking is based on the harvested root node relaxation values.

    Note: A unified list is used for all variables (x and y) because this VRPP
    formulation is a pure 0-1 MIP. While Guastaroba et al. (2017) suggests
    maintaining separate lists for binary and general integer variables, they
    are merged here as all discrete variables in this model are binary.
    """
    # Ensure Lazy Constraints are enabled for any solve with callbacks
    model.Params.LazyConstraints = 1

    # variables assumed binary from _setup_ks_model(use_binary_vars=True)

    # 2. Optimize with the root node callback (imported from KS solver)
    model.optimize(_root_node_callback)

    # 3. Extract relaxation values
    var_values = {var: val for var, val in zip(model._all_vars_list, model._node_rel, strict=False)}

    # 1.5 Compute a totally feasible heuristic route
    rng = random.Random()
    heuristic_routes = build_greedy_routes(
        dist_matrix=dist_matrix, wastes=wastes, capacity=capacity, R=R, C=C, mandatory_nodes=mandatory_nodes, rng=rng
    )

    heuristic_edges = set()
    heuristic_nodes = set()
    for route in heuristic_routes:
        if not route:
            continue
        heuristic_edges.add((0, route[0]))
        for i in range(len(route) - 1):
            heuristic_edges.add((route[i], route[i + 1]))
            heuristic_nodes.add(route[i])
        heuristic_edges.add((route[-1], 0))
        heuristic_nodes.add(route[-1])

    # 4. Extract variables and their harvested relaxation values
    all_vars = []
    for (i, j), var in x.items():
        val = var_values.get(var, 0.0)
        all_vars.append((var, val, "x", (i, j)))
    for i, var in y.items():
        val = var_values.get(var, 0.0)
        all_vars.append((var, val, "y", i))  # type: ignore[arg-type]

    # Sort logic: Root node relaxation value descending
    all_vars.sort(key=lambda item: item[1], reverse=True)

    # 5. Partition variables based on the sorted ranking
    lp_support_size = sum(1 for _, val, _, _ in all_vars if val > 1e-4)
    actual_kernel_size = max(initial_kernel_size, lp_support_size)

    kernel_vars = []
    remaining_vars = []

    # Include heuristic variables to guarantee ILP feasibility
    for v_obj, _, _, _ in all_vars[:actual_kernel_size]:
        kernel_vars.append(v_obj)

    for v_obj, _, vtype, idx in all_vars[actual_kernel_size:]:
        if (vtype == "x" and idx in heuristic_edges) or (vtype == "y" and idx in heuristic_nodes):
            kernel_vars.append(v_obj)
        else:
            remaining_vars.append(v_obj)

    # Organize remaining variables into sequential buckets
    buckets = [remaining_vars[i : i + bucket_size] for i in range(0, len(remaining_vars), bucket_size)]

    return kernel_vars, remaining_vars, buckets


def _get_feasible(model, remaining_vars, active_kernel):
    """
    Step 3: GETFEASIBLE Routine.
    Iteratively increases the size of kernel K until a feasible solution is found.
    """
    m_w = int(len(remaining_vars) * 0.30)
    current_rem_idx = 0
    while model.SolCount == 0 and current_rem_idx < len(remaining_vars):
        # Add next chunk of variables to kernel
        next_chunk = remaining_vars[current_rem_idx : current_rem_idx + m_w]
        for v in next_chunk:
            v.UB = 1
            active_kernel.add(v)
        current_rem_idx += m_w
        model.optimize(_dfj_subtour_elimination_callback)
    return current_rem_idx


def _assess_difficulty(model, t_mip_k, t_easy, epsilon):
    """
    Step 2: Difficulty Assessment.
    Classifies instance and applies HARD-mode constraints if needed.
    """
    t_hard = model.Params.TimeLimit
    classification = "NORMAL"
    if t_mip_k <= t_easy:
        classification = "EASY"
    elif t_mip_k >= t_hard:
        classification = "HARD"

    # Behavior for HARD: Fix binary variables with high root relaxation values
    if classification == "HARD":
        for v, rel_val in zip(model._all_vars_list, model._node_rel, strict=False):
            if rel_val > 1.0 - epsilon:
                v.LB = 1.0  # Permanently fix to 1
    return classification


def _solve_easy_iterations(model, remaining_vars, current_rem_idx, active_kernel, chunk_size, t_easy, x_h, best_obj):
    """
    Step 4a: EASY iteration logic.
    """
    while current_rem_idx < len(remaining_vars):
        end_idx = min(current_rem_idx + chunk_size, len(remaining_vars))
        chunk = remaining_vars[current_rem_idx:end_idx]
        for v in chunk:
            v.UB = 1

        model.setParam("TimeLimit", t_easy * 2)
        model.Params.Cutoff = best_obj + 1e-4

        solve_start = model.Runtime
        model.optimize(_dfj_subtour_elimination_callback)
        solve_duration = model.Runtime - solve_start

        if model.SolCount > 0:
            best_obj = model.ObjVal
            x_h.update({v: v.X for v in active_kernel.union(chunk)})

        if solve_duration > t_easy and model.SolCount > 0:
            for v in chunk:
                if v.X > 0.5:
                    active_kernel.add(v)
            break
        current_rem_idx = end_idx
    return current_rem_idx, best_obj


def _solve_rigorous_iterations(
    model, buckets, active_kernel, x_h, best_obj, max_buckets, time_limit, start_time, mip_limit_nodes
):
    """
    Step 4b: NORMAL/HARD iteration logic with bucket constraints.
    """
    for i, bucket in enumerate(buckets[:max_buckets]):
        elapsed = model.Runtime - start_time
        if elapsed > time_limit:
            break

        for v in bucket:
            v.UB = 1

        # Step 1: AKS Bucket Constraint
        k_zeros = [v for v in active_kernel if x_h.get(v, 0.0) < 0.5]
        bucket_constr = model.addConstr(gp.quicksum(bucket + k_zeros) >= 1, name="AKS_Bucket_Constraint")

        iter_time = max(5.0, (time_limit - elapsed) / max(1, max_buckets - i))
        model.setParam("TimeLimit", iter_time)
        model.setParam("NodeLimit", mip_limit_nodes)

        # Step 2: Objective Cutoff
        model.Params.Cutoff = best_obj + 1e-4

        model.optimize(_dfj_subtour_elimination_callback)

        if model.SolCount > 0 and model.ObjVal > best_obj + 1e-4:
            best_obj = model.ObjVal
            x_h.update({v: v.X for v in active_kernel.union(bucket)})
            active_kernel.update({v for v in bucket if v.X > 0.5})

        for v in bucket:
            if v not in active_kernel:
                v.UB = 0

        model.remove(bucket_constr)
    return best_obj


def _solve_aks_iterations(
    model: gp.Model,
    kernel_vars: List[gp.Var],
    remaining_vars: List[gp.Var],
    bucket_size: int,
    max_buckets: int,
    time_limit: float,
    mip_limit_nodes: int,
    t_easy: float = 10.0,
    epsilon: float = 0.1,
) -> Set[gp.Var]:
    """
    Execute the core Adaptive Kernel Search iterative improvement loop.
    Complies with Guastaroba et al. (2017) methodology.
    """
    active_kernel = set(kernel_vars)
    for v in remaining_vars:
        v.UB = 0

    best_obj = -float("inf")
    start_time = model.Runtime

    # PHASE 1: Initial Kernel Solve and potential GETFEASIBLE
    model.setParam("TimeLimit", max(10.0, time_limit * 0.2))
    model.optimize(_dfj_subtour_elimination_callback)

    current_rem_idx = 0
    if model.SolCount == 0:
        # Step 3: GETFEASIBLE Routine - Only if initial solve fails
        current_rem_idx = _get_feasible(model, remaining_vars, active_kernel)

    if model.SolCount == 0:
        return set()

    t_mip_k = model.Runtime - start_time
    best_obj = model.ObjVal
    x_h = {v: v.X for v in active_kernel}

    # PHASE 2: Difficulty Assessment
    classification = _assess_difficulty(model, t_mip_k, t_easy, epsilon)

    # PHASE 3: Iterative Bucket Solving
    if classification == "EASY":
        current_rem_idx, best_obj = _solve_easy_iterations(
            model, remaining_vars, current_rem_idx, active_kernel, bucket_size * 2, t_easy, x_h, best_obj
        )
    else:
        remaining_to_bucket = remaining_vars[current_rem_idx:]
        buckets = [remaining_to_bucket[i : i + bucket_size] for i in range(0, len(remaining_to_bucket), bucket_size)]
        best_obj = _solve_rigorous_iterations(
            model, buckets, active_kernel, x_h, best_obj, max_buckets, time_limit, start_time, mip_limit_nodes
        )

    return active_kernel


def run_adaptive_kernel_search_gurobi(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
    initial_kernel_size: int = 50,
    bucket_size: int = 20,
    max_buckets: int = 15,
    time_limit: float = 300.0,
    mip_limit_nodes: int = 10000,
    mip_gap: float = 0.01,
    t_easy: float = 10.0,
    epsilon: float = 0.1,
    seed: int = 42,
    env: Optional[gp.Env] = None,
    recorder: Optional[PolicyStateRecorder] = None,
) -> Tuple[List[int], float, float]:
    """
    Solve VRPP using Adaptive Kernel Search (AKS) per Guastaroba et al. (2017).

    Args:
        initial_kernel_size (int): Size of the starting variable pool.
        bucket_size (int): Size for search buckets.
        max_buckets (int): Limit on improvement attempts.
        time_limit (float): Total time budget for optimization.
        mip_limit_nodes (int): Node limit for internal Gurobi solves.
        mip_gap (float): Acceptable relative optimality gap.
        t_easy (float): Threshold for 'EASY' instance classification (seconds).
        epsilon (float): Tolerance for fixing variables in 'HARD' instances.
        seed (int): Random seed.
    """
    model = gp.Model("AKS_VRPP", env=env) if env else gp.Model("AKS_VRPP")
    model.setParam("OutputFlag", 0)
    model.setParam("Seed", seed)
    model.setParam("MIPGap", mip_gap)

    # Setup core formulation (ensure binary vars for root relaxation)
    x, y = _setup_ks_model(model, dist_matrix, wastes, capacity, R, C, mandatory_nodes, use_binary_vars=True)

    kernel_vars, remaining_vars, _ = _get_partitioned_vars_aks(
        model, x, y, initial_kernel_size, bucket_size, dist_matrix, wastes, capacity, R, C, mandatory_nodes
    )

    if not kernel_vars:
        return [0, 0], 0.0, 0.0

    used_vars = _solve_aks_iterations(
        model, kernel_vars, remaining_vars, bucket_size, max_buckets, time_limit, mip_limit_nodes, t_easy, epsilon
    )

    if not used_vars:
        return [0, 0], 0.0, 0.0

    for v in used_vars:
        v.UB = 1

    _set_mip_start(model, x, y, dist_matrix, wastes, capacity, R, C, mandatory_nodes)
    model.optimize(_dfj_subtour_elimination_callback)

    if model.SolCount == 0:
        return [0, 0], 0.0, 0.0

    tour, cost = _reconstruct_tour(len(dist_matrix), x, dist_matrix)
    if recorder:
        recorder.record(engine="adaptive_kernel_search", obj_val=model.ObjVal, cost=cost, solved=1)

    return tour, float(model.ObjVal), float(cost)
