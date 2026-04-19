import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import gurobipy as gp
import numpy as np

from logic.src.policies.route_construction.matheuristics.adaptive_kernel_search.aks import (
    _assess_difficulty,
    _get_feasible,
    _get_partitioned_vars_aks,
    _solve_easy_iterations,
    _solve_rigorous_iterations,
)
from logic.src.policies.route_construction.matheuristics.kernel_search.solver import (
    _dfj_subtour_elimination_callback,
    _reconstruct_tour,
    _set_mip_start,
    _setup_ks_model,
)
from logic.src.tracking.viz_mixin import PolicyStateRecorder

from .params import TPKSParams


@dataclass
class _PhaseIStats:
    """
    Per-variable statistics collected during Phase I.

    Used by Phase II to rank variables better than a fresh LP relaxation,
    since Phase I already explored the feasible region.
    """

    var_frequency: Dict[gp.Var, float] = field(default_factory=dict)
    # frequency[v] = fraction of Phase I MIP solutions in which v.X > 0.5
    var_obj_contribution: Dict[gp.Var, float] = field(default_factory=dict)
    # obj_contribution[v] = average objective improvement when v was added to kernel
    phase1_best_obj: float = -float("inf")
    phase1_used_vars: Set[gp.Var] = field(default_factory=set)


def _run_phase1(  # noqa: C901
    model: gp.Model,
    x: Dict,
    y: Dict,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
    params: TPKSParams,
    phase1_time: float,
) -> _PhaseIStats:
    """
    Phase I: Feasibility-focused search.

    Goal: find a high-quality feasible solution as quickly as possible,
    while recording per-variable statistics to inform Phase II.

    Steps:
    1. Partition variables using `_get_partitioned_vars_aks` with
       `initial_kernel_size = params.phase1_kernel_size`.
       This runs the root LP relaxation (with fractional subtour separation)
       to rank variables.

    2. Solve the Phase I kernel MIP (time budget = phase1_time * 0.5).
       If SolCount == 0 after this solve, call `_get_feasible` (imported from
       aks.py) to iteratively expand the kernel until a feasible solution is
       found. This is the core Phase I contribution: guaranteeing a feasible
       start point even when the LP ranking is misleading.

    3. Record statistics: after each MIP solve (kernel + each bucket added
       during GETFEASIBLE), for every variable v with v.X > 0.5 increment
       `var_frequency[v]` by 1/n_solves and record objective contribution.

    4. Apply a single pass of rapid bucket expansion (bucket size =
       params.phase1_bucket_size) using the remaining phase1_time budget,
       calling `_dfj_subtour_elimination_callback` for each bucket solve.
       Promote variables to the kernel if they appear in improving solutions.
       Continue recording frequency statistics.

    Returns:
        _PhaseIStats populated with frequency/contribution data and the
        best solution found in Phase I.
    """
    stats = _PhaseIStats()
    start = time.perf_counter()
    n_solves = 0

    # 1. Partition using AKS-style root LP (imports _get_partitioned_vars_aks)
    kernel_vars, remaining_vars, _ = _get_partitioned_vars_aks(
        model,
        x,
        y,
        initial_kernel_size=params.phase1_kernel_size,
        bucket_size=params.phase1_bucket_size,
        dist_matrix=dist_matrix,
        wastes=wastes,
        capacity=capacity,
        R=R,
        C=C,
        mandatory_nodes=mandatory_nodes,
    )

    active_kernel: Set[gp.Var] = set(kernel_vars)
    for v in remaining_vars:
        v.UB = 0

    # 2. Initial kernel solve
    phase1_kernel_time = phase1_time * 0.5
    model.setParam("TimeLimit", phase1_kernel_time)
    model.setParam("NodeLimit", params.phase1_mip_node_limit)
    model.optimize(_dfj_subtour_elimination_callback)

    if model.SolCount > 0:
        n_solves += 1
        stats.phase1_best_obj = model.ObjVal
        for v in active_kernel:
            if v.X > 0.5:
                stats.var_frequency[v] = stats.var_frequency.get(v, 0.0) + 1.0
                stats.phase1_used_vars.add(v)

    # If still infeasible, call _get_feasible to expand until feasible
    if model.SolCount == 0:
        _get_feasible(model, remaining_vars, active_kernel)
        if model.SolCount > 0:
            n_solves += 1
            stats.phase1_best_obj = model.ObjVal
            for v in active_kernel:
                if v.X > 0.5:
                    stats.var_frequency[v] = stats.var_frequency.get(v, 0.0) + 1.0
                    stats.phase1_used_vars.add(v)

    # 3. Rapid bucket expansion with remaining Phase I budget
    elapsed = time.perf_counter() - start
    bucket_budget = max(0.0, phase1_time - elapsed)
    bucket_start = time.perf_counter()
    prev_obj = stats.phase1_best_obj

    remaining_unfixed = [v for v in remaining_vars if v.UB < 0.5]
    buckets_p1 = [
        remaining_unfixed[i : i + params.phase1_bucket_size]
        for i in range(0, len(remaining_unfixed), params.phase1_bucket_size)
    ]

    for bucket in buckets_p1:
        if time.perf_counter() - bucket_start > bucket_budget:
            break

        for v in bucket:
            v.UB = 1

        b_constr = model.addConstr(gp.quicksum(bucket) >= 1, name="TPKS_P1_Bucket")
        model.setParam("TimeLimit", max(5.0, bucket_budget / max(1, len(buckets_p1))))
        model.Params.Cutoff = stats.phase1_best_obj + 1e-4
        model.optimize(_dfj_subtour_elimination_callback)

        if model.SolCount > 0 and model.ObjVal > stats.phase1_best_obj + 1e-4:
            n_solves += 1
            delta_obj = model.ObjVal - prev_obj
            for v in bucket:
                if v.X > 0.5:
                    stats.var_frequency[v] = stats.var_frequency.get(v, 0.0) + 1.0
                    stats.var_obj_contribution[v] = stats.var_obj_contribution.get(v, 0.0) + delta_obj
                    active_kernel.add(v)
                    stats.phase1_used_vars.add(v)
            stats.phase1_best_obj = model.ObjVal
            prev_obj = model.ObjVal

        for v in bucket:
            if v not in active_kernel:
                v.UB = 0

        model.remove(b_constr)

    # Normalise frequency by number of solves
    if n_solves > 0:
        for v in stats.var_frequency:
            stats.var_frequency[v] /= n_solves

    return stats


def _build_phase2_kernel(
    x: Dict,
    y: Dict,
    stats: _PhaseIStats,
    params: TPKSParams,
) -> Tuple[List[gp.Var], List[gp.Var]]:
    """
    Phase II: Re-rank all variables using Phase I statistics.

    Scoring function per variable v:
        score(v) = α · frequency(v) + β · obj_contribution(v) / normaliser
    where α = 0.6, β = 0.4.

    The top `initial_kernel_size` variables (union with `phase1_used_vars`)
    form the Phase II kernel. The remainder are sorted into buckets.

    Returns:
        (phase2_kernel_vars, phase2_remaining_vars)
    """
    all_vars_scored: List[Tuple[gp.Var, float]] = []
    max_contrib = max(list(stats.var_obj_contribution.values()) + [1.0], default=1.0)
    if max_contrib <= 0:
        max_contrib = 1.0

    for (_i, _j), v in x.items():
        freq = stats.var_frequency.get(v, 0.0)
        contrib = stats.var_obj_contribution.get(v, 0.0) / max_contrib
        all_vars_scored.append((v, 0.6 * freq + 0.4 * contrib))

    for _i, v in y.items():
        freq = stats.var_frequency.get(v, 0.0)
        contrib = stats.var_obj_contribution.get(v, 0.0) / max_contrib
        all_vars_scored.append((v, 0.6 * freq + 0.4 * contrib))

    all_vars_scored.sort(key=lambda t: t[1], reverse=True)

    # Always include Phase I used vars in the Phase II kernel
    phase2_kernel_set: Set[gp.Var] = set(stats.phase1_used_vars)
    for v, _ in all_vars_scored[: params.initial_kernel_size]:
        phase2_kernel_set.add(v)

    phase2_kernel = list(phase2_kernel_set)
    phase2_remaining = [v for v, _ in all_vars_scored if v not in phase2_kernel_set]

    return phase2_kernel, phase2_remaining


def run_tpks_gurobi(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
    params: Optional[TPKSParams] = None,
    env: Optional[gp.Env] = None,
    recorder: Optional[PolicyStateRecorder] = None,
) -> Tuple[List[int], float, float]:
    """
    Solve VRPP using Two-Phase Kernel Search (TPKS).

    Returns:
        (tour, obj_val, routing_cost) — same contract as run_kernel_search_gurobi.
    """
    if params is None:
        params = TPKSParams()

    model = gp.Model("TPKS_VRPP", env=env) if env else gp.Model("TPKS_VRPP")
    model.setParam("OutputFlag", 0)
    model.setParam("Seed", params.seed)
    model.setParam("MIPGap", params.mip_gap)
    model.Params.LazyConstraints = 1

    # Build MIP model (binary vars for root LP; same as KS/AKS)
    x, y = _setup_ks_model(
        model,
        dist_matrix,
        wastes,
        capacity,
        R,
        C,
        mandatory_nodes,
        use_binary_vars=True,
    )

    # Warm-start with greedy heuristic
    _set_mip_start(model, x, y, dist_matrix, wastes, capacity, R, C, mandatory_nodes)

    # ---------------------------------------------------------------
    # PHASE I — Feasibility + statistics collection
    # ---------------------------------------------------------------
    phase1_time = params.time_limit * params.phase1_time_fraction
    stats = _run_phase1(
        model,
        x,
        y,
        dist_matrix,
        wastes,
        capacity,
        R,
        C,
        mandatory_nodes,
        params,
        phase1_time,
    )

    if stats.phase1_best_obj == -float("inf"):
        # Phase I completely failed — return greedy fallback
        tour, cost = _reconstruct_tour(len(dist_matrix), x, dist_matrix)
        return tour, 0.0, cost

    # ---------------------------------------------------------------
    # PHASE II — Quality improvement using Phase I statistics
    # ---------------------------------------------------------------
    phase2_time = params.time_limit - phase1_time
    phase2_kernel, phase2_remaining = _build_phase2_kernel(x, y, stats, params)

    # Reset all bounds; only Phase II kernel is active
    for v in phase2_remaining:
        v.UB = 0
    for v in phase2_kernel:
        v.UB = 1

    # Difficulty assessment (reuse AKS logic)
    start_p2 = time.perf_counter()
    model.setParam("TimeLimit", max(5.0, phase2_time * 0.2))
    model.setParam("NodeLimit", params.mip_limit_nodes)
    model.Params.Cutoff = stats.phase1_best_obj + 1e-4
    model.optimize(_dfj_subtour_elimination_callback)

    t_mip_k = time.perf_counter() - start_p2
    best_obj = model.ObjVal if model.SolCount > 0 else stats.phase1_best_obj
    x_h: Dict[gp.Var, float] = {v: v.X for v in phase2_kernel if model.SolCount > 0 and v.X > 0.5}

    classification = _assess_difficulty(model, t_mip_k, params.t_easy, params.epsilon)

    # Adaptive bucket sizing based on difficulty classification
    bucket_size_p2 = params.phase2_bucket_size_easy if classification == "EASY" else params.phase2_bucket_size_normal

    buckets_p2 = [phase2_remaining[i : i + bucket_size_p2] for i in range(0, len(phase2_remaining), bucket_size_p2)]
    buckets_p2 = buckets_p2[: params.max_buckets]
    if classification == "EASY":
        _, best_obj = _solve_easy_iterations(
            model, phase2_remaining, 0, set(phase2_kernel), bucket_size_p2, params.t_easy, x_h, best_obj
        )
    else:
        active_kernel_set = set(phase2_kernel)
        best_obj = _solve_rigorous_iterations(
            model,
            buckets_p2,
            active_kernel_set,
            x_h,
            best_obj,
            params.max_buckets,
            phase2_time,
            start_p2,
            params.mip_limit_nodes,
        )

    # ---------------------------------------------------------------
    # Final polishing solve over the promoted kernel
    # ---------------------------------------------------------------
    promoted = {v for v in (list(phase2_kernel) + list(phase2_remaining)) if x_h.get(v, 0.0) > 0.5}
    for v in promoted:
        v.UB = 1

    remaining_budget = params.time_limit - (time.perf_counter() - start_p2) - phase1_time
    if remaining_budget > 2.0:
        model.setParam("TimeLimit", remaining_budget)
        model.Params.Cutoff = best_obj + 1e-4
        model.optimize(_dfj_subtour_elimination_callback)

    if model.SolCount == 0:
        return [0, 0], 0.0, 0.0

    tour, cost = _reconstruct_tour(len(dist_matrix), x, dist_matrix)
    obj_val = float(model.ObjVal)

    if recorder:
        recorder.record(
            engine="tpks",
            phase1_best=stats.phase1_best_obj,
            phase2_best=obj_val,
            cost=cost,
        )

    return tour, obj_val, cost
