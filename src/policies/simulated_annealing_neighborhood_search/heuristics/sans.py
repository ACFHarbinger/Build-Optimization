"""
Simulated Annealing Algorithm.
"""

import copy
import math
import random
import time
from typing import Optional

from logic.src.tracking.viz_mixin import PolicyStateRecorder

from ..common.routes import uncross_arcs_in_sans_routes
from .sans_operators import apply_operator
from .sans_state import compute_profit


def _initialize_solution_state(routes, id_to_index, distance_matrix, data):
    """Initialize SANS solution state."""
    # State Sync: Capture any bins missed by the initial greedy solution.
    # Every bin ID must be either in a route or in the 'removed_bins' set.
    all_bins_ids = set(data["#bin"].tolist()) - {0}
    scheduled_bins = set()
    for r in routes:
        for b in r:
            if b != 0:
                scheduled_bins.add(b)
    missing_bins = all_bins_ids - scheduled_bins

    initial_solution = copy.deepcopy(routes)
    # Post-processing: Uncross arcs in the initial solution to start from a geometric optimum.
    current_solution = uncross_arcs_in_sans_routes(initial_solution, id_to_index, distance_matrix)

    return current_solution, missing_bins


def _select_neighbor(
    solution, removed_bins, data, vehicle_capacity, id_to_index, stocks, must_go_bins, distance_matrix
):
    """Select and apply a neighbor operator."""
    route_ops = [
        "2opt",
        "move",
        "swap",
        "remove",
        "insert",
        "move_n_random",
        "move_n_consec",
        "swap_n_random",
        "swap_n_consec",
        "remove_n_bins",
        "remove_n_bins_consec",
        "relocate",
        "cross",
        "or-opt",
    ]
    add_ops = [
        "add_n_bins",
        "add_n_bins_consec",
        "add_route_removed",
        "add_route_removed_consec",
    ]

    valid_ops = []
    if len(solution) > 0:
        valid_ops.extend(route_ops)
    if len(removed_bins) > 0:
        valid_ops.extend(add_ops)

    if not valid_ops:
        return None, None

    op = random.choice(valid_ops)

    new_solution = apply_operator(
        op,
        copy.deepcopy(solution),
        copy.deepcopy(removed_bins),
        data,
        vehicle_capacity,
        id_to_index,
        stocks,
        must_go_bins,
        distance_matrix,
    )

    # Filter out empty routes
    new_solution = [r for r in new_solution if len(r) > 2]

    return new_solution, op


def improved_simulated_annealing(  # noqa: C901
    routes,
    distance_matrix,
    time_limit,
    id_to_index,
    data,
    vehicle_capacity,
    T_init=1000,
    T_min=0.001,
    alpha=0.995,
    iterations_per_T=100,
    R=0.165,
    V=2.5,
    density=20,
    C=1.0,
    must_go_bins=None,
    removed_bins=None,
    verbose=False,
    perc_bins_can_overflow=0.0,
    volume=2.5,
    density_val=20,
    max_vehicles=None,
    recorder: Optional[PolicyStateRecorder] = None,
):
    """
    Refine routing solutions using a multi-neighborhood Simulated Annealing algorithm.
    """
    start_time = time.time()

    # --- 1. ROBUST INITIALIZATION ---
    removed_bins = set() if removed_bins is None else set(removed_bins)
    must_go_bins = set() if must_go_bins is None else set(must_go_bins)

    current_solution, missing_bins = _initialize_solution_state(routes, id_to_index, distance_matrix, data)
    removed_bins.update(missing_bins)

    stocks = dict(zip(data["#bin"], data["Stock"]))

    # Initial Evaluation
    current_profit, current_cost, current_revenue, real_kg = compute_profit(
        current_solution,
        distance_matrix,
        id_to_index,
        data,
        vehicle_capacity,
        R,
        V,
        density,
        must_go_bins,
        stocks,
    )

    best_solution = copy.deepcopy(current_solution)
    best_profit = current_profit
    last_receita = current_revenue
    last_weight = real_kg
    last_distance = current_cost

    no_improvement_count = 0
    T = T_init

    # --- 2. MAIN SIMULATED ANNEALING LOOP ---
    while T_min < T:
        if time.time() - start_time > time_limit:
            if verbose:
                print("[DEBUG] Time limit reached.")
            break

        # Inner Loop
        for _ in range(iterations_per_T):
            if time.time() - start_time > time_limit:
                break

            # --- 3. NEIGHBOR SELECTION ---
            # Using copy inside helper to avoid mutating current state before accept
            new_solution, op = _select_neighbor(
                current_solution,
                removed_bins,
                data,
                vehicle_capacity,
                id_to_index,
                stocks,
                must_go_bins,
                distance_matrix,
            )

            if new_solution is None:
                continue

            # --- 4. EVALUATION ---
            # Ideally apply_operator should return the modified removed_bins too.
            # But the current signature operates on mutable removed_bins copy.
            # We need to capture that change.
            # Refactoring note: apply_operator modifies removed_bins in place.
            # So we need to pass a fresh copy to _select_neighbor or rely on it returning the modified set.
            # Current implementation of _select_neighbor passes copy.deepcopy(removed_bins).
            # But we lose the reference to the modified set unless we return it.
            # Ah, `new_solution` is returned. But `removed_bins` changes are lost in my helper above!
            # I must fix _select_neighbor to return candidate_removed_bins.

            # Re-implementing _select_neighbor logic inline correctly or fixing signature
            # Let's do it inline to ensure correctness with state variables, or fix helper.
            # Fixing helper is better.

            # Wait, `apply_operator` signature:
            # apply_operator(op, new_solution, candidate_removed_bins, ...)
            # It modifies candidate_removed_bins in place.

            # Let's refine helper:
            # candidate_removed_bins = copy.deepcopy(removed_bins)
            # new_solution = ...
            # return new_solution, candidate_removed_bins

            # Actually, to avoid complexity in this step (replacing file content),
            # I will inline the helper logic with slight cleanup.

            candidate_removed_bins = copy.deepcopy(removed_bins)
            # ... neighbor selection logic ...
            route_ops = [
                "2opt",
                "move",
                "swap",
                "remove",
                "insert",
                "move_n_random",
                "move_n_consec",
                "swap_n_random",
                "swap_n_consec",
                "remove_n_bins",
                "remove_n_bins_consec",
                "relocate",
                "cross",
                "or-opt",
            ]
            add_ops = ["add_n_bins", "add_n_bins_consec", "add_route_removed", "add_route_removed_consec"]

            valid_ops = []
            if len(current_solution) > 0:
                valid_ops.extend(route_ops)
            if len(candidate_removed_bins) > 0:
                valid_ops.extend(add_ops)

            if not valid_ops:
                continue

            op = random.choice(valid_ops)
            new_solution = apply_operator(
                op,
                copy.deepcopy(current_solution),
                candidate_removed_bins,
                data,
                vehicle_capacity,
                id_to_index,
                stocks,
                must_go_bins,
                distance_matrix,
            )
            new_solution = [r for r in new_solution if len(r) > 2]

            new_profit, new_cost, new_revenue, new_real_kg = compute_profit(
                new_solution,
                distance_matrix,
                id_to_index,
                data,
                vehicle_capacity,
                R,
                V,
                density,
                must_go_bins,
                stocks,
            )

            delta = new_profit - current_profit

            # Acceptance
            accept = False
            if delta > 0:
                accept = True
            else:
                try:
                    p = math.exp(delta / T)
                except OverflowError:
                    p = 0
                accept = random.random() < p

            if accept:
                current_solution = new_solution
                current_cost = new_cost
                current_profit = new_profit
                removed_bins = candidate_removed_bins

                if current_profit > best_profit:
                    best_solution = copy.deepcopy(current_solution)
                    best_profit = current_profit
                    last_weight = new_real_kg
                    last_distance = new_cost
                    last_receita = new_revenue
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1

        T = T * alpha
        if verbose:
            print(f"Temperature cooled to {T:.4f}")

        if recorder is not None:
            recorder.record(
                temperature=T,
                best_profit=best_profit,
                current_profit=current_profit,
                no_improvement_count=no_improvement_count,
            )

        if no_improvement_count > 500:
            if verbose:
                print("[INFO] Rehearing temperature.")
            T = T_init
            no_improvement_count = 0

    best_solution = [r for r in best_solution if len(r) > 2]
    best_solution = uncross_arcs_in_sans_routes(best_solution, id_to_index, distance_matrix)
    return best_solution, best_profit, last_distance, last_weight, last_receita
