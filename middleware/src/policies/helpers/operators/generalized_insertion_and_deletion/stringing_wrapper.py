"""
Global wrapper for Stringing operators as a general repair heuristic.

Provides `stringing_insertion` and `stringing_profit_insertion` wrappers
that mirror `greedy_insertion`, capable of iterating over unassigned nodes
(with `expand_pool` support) and inserting them using GENIUS Stringing moves.

Supports both:
1. Deterministic p-neighborhood search (strict adherence to Gendreau et al. 1992)
2. Randomized sampling mode (for faster approximate search)
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators.generalized_insertion_and_deletion.stringing_i import (
    apply_type_i_s,
    apply_type_i_s_profit,
)
from logic.src.policies.helpers.operators.generalized_insertion_and_deletion.stringing_ii import (
    apply_type_ii_s,
    apply_type_ii_s_profit,
)
from logic.src.policies.helpers.operators.generalized_insertion_and_deletion.stringing_iii import (
    apply_type_iii_s,
    apply_type_iii_s_profit,
)
from logic.src.policies.helpers.operators.generalized_insertion_and_deletion.stringing_iv import (
    apply_type_iv_s,
    apply_type_iv_s_profit,
)
from logic.src.utils.policy.neighborhood import get_p_neighborhood


def _evaluate_routes(routes: List[List[int]], dist_matrix: np.ndarray) -> float:
    """Evaluate total route cost."""
    cost = 0.0
    for rt in routes:
        if not rt:
            continue
        d = dist_matrix[0][rt[0]]
        for idx in range(len(rt) - 1):
            d += dist_matrix[rt[idx]][rt[idx + 1]]
        d += dist_matrix[rt[-1]][0]
        cost += d
    return -cost


def _apply_stringing_op(
    route: List[int],
    node: int,
    string_type: int,
    params: Tuple,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    profit_mode: bool,
    current_load: float,
) -> Tuple[List[int], float]:
    """Helper to apply stringing operation and get new route and immediate profit val."""
    if string_type in (2, 4):
        i, j, k, l = params
        if profit_mode:
            if string_type == 2:
                return apply_type_ii_s_profit(
                    route, node, i, j, k, l, dist_matrix, wastes, capacity, R, C, current_load
                )
            return apply_type_iv_s_profit(route, node, i, j, k, l, dist_matrix, wastes, capacity, R, C, current_load)
        new_route = (
            apply_type_ii_s(route, node, i, j, k, l, current_load)
            if string_type == 2
            else apply_type_iv_s(route, node, i, j, k, l, current_load)
        )
        return new_route, 0.0

    i, j, k = params
    if profit_mode:
        if string_type == 1:
            return apply_type_i_s_profit(route, node, i, j, k, dist_matrix, wastes, capacity, R, C, current_load)
        return apply_type_iii_s_profit(route, node, i, j, k, dist_matrix, wastes, capacity, R, C, current_load)
    new_route = (
        apply_type_i_s(route, node, i, j, k, current_load)
        if string_type == 1
        else apply_type_iii_s(route, node, i, j, k, current_load)
    )
    return new_route, 0.0


def _try_string_insertion(
    routes: List[List[int]],
    node: int,
    r_idx: int,
    string_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    rng: random.Random,
    current_load: float,
    profit_mode: bool = False,
) -> Optional[Tuple[List[List[int]], float]]:
    """Try inserting a node using the specified stringing operator, returning the best result."""
    route = routes[r_idx]
    if len(route) < 3:
        return None

    best_routes = None
    best_value = float("-inf")

    for _ in range(min(5, len(route))):
        valid_positions = list(range(len(route)))
        if len(valid_positions) < (4 if string_type in (2, 4) else 3):
            continue

        try:
            i = rng.choice(valid_positions)
            valid_j = [p for p in valid_positions if p != i]
            if not valid_j:
                continue
            j = rng.choice(valid_j)
            valid_k = [p for p in valid_positions if p not in (i, j)]
            if not valid_k:
                continue
            k = rng.choice(valid_k)

            if string_type in (2, 4):
                valid_l = [p for p in valid_positions if p not in (i, j, k)]
                if not valid_l:
                    continue
                l = rng.choice(valid_l)
                params: Tuple[int, ...] = (i, j, k, l)
            else:
                params = (i, j, k)  # type: ignore[assignment]

            new_route, val = _apply_stringing_op(
                route, node, string_type, params, dist_matrix, wastes, capacity, R, C, profit_mode, current_load
            )

            # Validate insertion (capacity and existence)
            # Optimization: since we do early pruning for capacity, we don't need the O(N) sum() here
            # anymore, stringing_wrapper enforces the check before trying the topological strings.
            if node not in new_route:
                continue

            test_routes = [list(r) for r in routes]
            test_routes[r_idx] = new_route

            if not profit_mode:
                val = _evaluate_routes(test_routes, dist_matrix)

            if val > best_value:
                best_value = val
                best_routes = test_routes

        except Exception:
            continue

    return (best_routes, best_value) if best_routes is not None else None


def _try_string_insertion_deterministic(  # noqa: C901
    routes: List[List[int]],
    node: int,
    r_idx: int,
    string_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    current_load: float,
    neighborhood_size: int,
    profit_mode: bool = False,
) -> Optional[Tuple[List[List[int]], float]]:
    """Try inserting a node using deterministic p-neighborhood search (bidirectional)."""
    route = routes[r_idx]
    if len(route) < 3:
        return None

    best_routes = None
    best_value = float("-inf")

    # Evaluate both orientations
    for target_route in [route, route[::-1]]:
        # Get p-neighborhood of the target node (u)
        p_neighbors_u = get_p_neighborhood(node, target_route, dist_matrix, neighborhood_size)

        for v_i in p_neighbors_u:
            i = target_route.index(v_i)
            v_ip1 = target_route[(i + 1) % len(target_route)]
            p_neighbors_ip1 = get_p_neighborhood(v_ip1, target_route, dist_matrix, neighborhood_size)

            for v_j in p_neighbors_u:
                j = target_route.index(v_j)
                if j == i:
                    continue
                v_jp1 = target_route[(j + 1) % len(target_route)]
                p_neighbors_jp1 = get_p_neighborhood(v_jp1, target_route, dist_matrix, neighborhood_size)

                try:
                    if string_type in (2, 4):
                        # Phase 3: Decouple neighborhoods
                        # v_k from N_p(v_{i+1})
                        for v_k in p_neighbors_ip1:
                            k = target_route.index(v_k)
                            if k in (i, j):
                                continue
                            # v_l from N_p(v_{j+1})
                            for v_l in p_neighbors_jp1:
                                l = target_route.index(v_l)
                                if l in (i, j, k):
                                    continue

                                params: Tuple[int, ...] = (i, j, k, l)
                                new_route, val = _apply_stringing_op(
                                    target_route,
                                    node,
                                    string_type,
                                    params,
                                    dist_matrix,
                                    wastes,
                                    capacity,
                                    R,
                                    C,
                                    profit_mode,
                                    current_load,
                                )
                                if node in new_route:
                                    test_routes = [list(r) for r in routes]
                                    test_routes[r_idx] = new_route
                                    if not profit_mode:
                                        val = _evaluate_routes(test_routes, dist_matrix)
                                    if val > best_value:
                                        best_value, best_routes = val, test_routes
                    else:
                        # Types I and III: v_k \in N_p(v_{i+1})
                        for v_k in p_neighbors_ip1:
                            k = target_route.index(v_k)
                            if k in (i, j):
                                continue

                            params = (i, j, k)  # type: ignore[assignment]
                            new_route, val = _apply_stringing_op(
                                target_route,
                                node,
                                string_type,
                                params,
                                dist_matrix,
                                wastes,
                                capacity,
                                R,
                                C,
                                profit_mode,
                                current_load,
                            )
                            if node in new_route:
                                test_routes = [list(r) for r in routes]
                                test_routes[r_idx] = new_route
                                if not profit_mode:
                                    val = _evaluate_routes(test_routes, dist_matrix)
                                if val > best_value:
                                    best_value, best_routes = val, test_routes
                except Exception:
                    continue

    return (best_routes, best_value) if best_routes is not None else None


def stringing_insertion_wrapper(  # noqa: C901
    routes: List[List[int]],
    removed_nodes: List[int],
    string_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float = 0.0,
    C: float = 1.0,
    mandatory_nodes: Optional[List[int]] = None,
    rng: Optional[random.Random] = None,
    profit_mode: bool = False,
    expand_pool: bool = False,
    use_alns_fallback: bool = False,
    random_us_sampling: bool = True,
    neighborhood_size: int = 5,
) -> List[List[int]]:
    """
    Core stringing insertion logic with conditional search mode.

    Args:
        random_us_sampling: If True, use random sampling; if False, use deterministic p-neighborhoods.
        neighborhood_size: Size of p-neighborhood (for deterministic mode).

    Fallback to greedy insertion if stringing fails (e.g., node cannot fit, route too short).
    """
    if rng is None:
        rng = random.Random()

    from logic.src.policies.helpers.operators.recreate_repair.greedy import (
        greedy_insertion,
        greedy_profit_insertion,
    )

    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()

    # Establish pool of unassigned candidates
    if expand_pool:
        visited = {n for r in routes for n in r}
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = sorted(list(removed_nodes))

    # Iterate until pool is resolved
    while unassigned:
        best_routes = None
        best_value = -float("inf")
        best_node = -1

        # O(N) Pre-calculate Route Capacities
        route_loads = [sum(wastes.get(n, 0) for n in r) for r in routes]

        # Try inserting every available unassigned node into every route
        for node in unassigned:
            is_mandatory = node in mandatory_nodes_set
            node_waste = wastes.get(node, 0)

            # Evaluate opening a brand new route `[0, node, 0]`
            if profit_mode:
                cost_new = dist_matrix[0, node] + dist_matrix[node, 0]
                val_new = (node_waste * R) - (cost_new * C)
                effective_val_new = val_new + (1e9 if is_mandatory else 0)

                # Reject negative pure-profit insertions unless mandatory
                if (is_mandatory or val_new >= -1e-4) and effective_val_new > best_value:
                    best_value = effective_val_new
                    test_routes_new = [list(r) for r in routes]
                    test_routes_new.append([node])
                    best_routes = test_routes_new
                    best_node = node
            else:
                test_routes_new = [list(r) for r in routes]
                test_routes_new.append([node])
                val_new = _evaluate_routes(test_routes_new, dist_matrix)
                if val_new > best_value:
                    best_value = val_new
                    best_routes = test_routes_new
                    best_node = node

            for r_idx in range(len(routes)):
                # Early Pruning: Capacity Constraint Check BEFORE topological iterations
                current_load = route_loads[r_idx]
                if current_load + node_waste > capacity:
                    continue

                # Choose search strategy based on random_us_sampling flag
                if random_us_sampling:
                    result = _try_string_insertion(
                        routes,
                        node,
                        r_idx,
                        string_type,
                        dist_matrix,
                        wastes,
                        capacity,
                        R,
                        C,
                        rng,
                        current_load,
                        profit_mode,
                    )
                else:
                    result = _try_string_insertion_deterministic(
                        routes,
                        node,
                        r_idx,
                        string_type,
                        dist_matrix,
                        wastes,
                        capacity,
                        R,
                        C,
                        current_load,
                        neighborhood_size,
                        profit_mode,
                    )

                if result:
                    test_routes, val = result

                    if profit_mode:
                        # Bias score slightly for mandatory nodes in VRPP mode
                        effective_val = val + (1e9 if is_mandatory else 0)
                        # Reject negative pure-profit insertions unless mandatory
                        if not is_mandatory and val < -1e-4:
                            continue

                        if effective_val > best_value:
                            best_value = effective_val
                            best_routes = test_routes
                            best_node = node
                    else:
                        if val > best_value:
                            best_value = val
                            best_routes = test_routes
                            best_node = node

        # Apply best found insertion
        if best_node != -1 and best_routes is not None:
            routes = best_routes
            unassigned.remove(best_node)
        else:
            if not use_alns_fallback:
                # If unable to string properly, and fallback is disabled, abandon unassigned nodes
                break

            # Fallback for remaining nodes that couldn't be strung (too small routes, complex constraints)
            if profit_mode:
                routes = greedy_profit_insertion(
                    routes,
                    unassigned,
                    dist_matrix,
                    wastes,
                    capacity,
                    R,
                    C,
                    mandatory_nodes,
                    expand_pool=False,  # Already isolated to unassigned pool
                )
            else:
                routes = greedy_insertion(
                    routes,
                    unassigned,
                    dist_matrix,
                    wastes,
                    capacity,
                    mandatory_nodes=mandatory_nodes,
                    expand_pool=False,
                )
            break  # Greedy fallback completely drains the unassigned array

    return routes


# ==== Public Wrapper API ====


def stringing_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    string_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    mandatory_nodes: Optional[List[int]] = None,
    rng: Optional[random.Random] = None,
    expand_pool: bool = False,
    use_alns_fallback: bool = False,
    random_us_sampling: bool = True,
    neighborhood_size: int = 5,
) -> List[List[int]]:
    """Standard CVRP Stringing Repair."""
    return stringing_insertion_wrapper(
        routes,
        removed_nodes,
        string_type,
        dist_matrix,
        wastes,
        capacity,
        mandatory_nodes=mandatory_nodes,
        rng=rng,
        profit_mode=False,
        expand_pool=expand_pool,
        use_alns_fallback=use_alns_fallback,
        random_us_sampling=random_us_sampling,
        neighborhood_size=neighborhood_size,
    )


def stringing_profit_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    string_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    rng: Optional[random.Random] = None,
    expand_pool: bool = False,
    use_alns_fallback: bool = False,
    random_us_sampling: bool = True,
    neighborhood_size: int = 5,
) -> List[List[int]]:
    """VRPP Cost-Aware Stringing Repair."""
    # Feedback implementation: Keep Variants I, II, and III purely structural
    profit_mode = string_type not in (1, 2, 3)
    return stringing_insertion_wrapper(
        routes,
        removed_nodes,
        string_type,
        dist_matrix,
        wastes,
        capacity,
        R,
        C,
        mandatory_nodes=mandatory_nodes,
        rng=rng,
        profit_mode=profit_mode,
        expand_pool=expand_pool,
        use_alns_fallback=use_alns_fallback,
        random_us_sampling=random_us_sampling,
        neighborhood_size=neighborhood_size,
    )
