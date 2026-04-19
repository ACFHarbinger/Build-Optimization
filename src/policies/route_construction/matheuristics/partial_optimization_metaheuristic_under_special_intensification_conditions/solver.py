"""
Core implementation of POPMUSIC (Partial Optimization Metaheuristic Under Special Intensification Conditions).

POPMUSIC is a matheuristic framework that decomposes a large combinatorial
optimization problem into subproblems (sets of routes) and iteratively
optimizes them.

Reference:
    Taillard, É., & Voss, S. (2002). "POPMUSIC: a metaheuristic for routing
    problems". Metaheuristics: theory, research and applications, 185-200.
"""

import inspect
import warnings
from collections import deque
from random import Random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from logic.src.configs.policies.alns import ALNSConfig
from logic.src.configs.policies.hgs import HGSConfig
from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes
from logic.src.policies.helpers.operators.solution_initialization.nearest_neighbor_si import build_nn_routes
from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns import ALNSSolver
from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params import ALNSParams
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs import HGSSolver
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.params import HGSParams
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.split import LinearSplit
from logic.src.policies.route_construction.other_algorithms.travelling_salesman_problem.tsp import (
    find_route,
    get_route_cost,
)


def run_popmusic(  # noqa: C901
    coords: pd.DataFrame,
    mandatory: List[int],
    distance_matrix: np.ndarray,
    n_vehicles: int,
    subproblem_size: int = 3,
    max_iterations: Optional[int] = None,
    base_solver: str = "fast_tsp",
    base_solver_config: Optional[Any] = None,
    cluster_solver: str = "fast_tsp",
    cluster_solver_config: Optional[Any] = None,
    initial_solver: str = "nearest_neighbor",
    seed: int = 42,
    wastes: Optional[Dict[int, float]] = None,
    capacity: float = 1.0e9,
    R: float = 1.0,
    C: float = 0.0,
    vrpp: bool = True,
    profit_aware_operators: bool = False,
    k_prox: int = 10,
    seed_strategy: str = "lifo",
) -> Tuple[List[List[int]], float, float, Dict[str, Any]]:
    """
    Solve VRPP using POPMUSIC matheuristic.

    Args:
        coords: Node coordinates.
        mandatory: List of bin indices to collect.
        distance_matrix: Distances between nodes.
        n_vehicles: Number of vehicles available.
        subproblem_size: Total number of routes per subproblem, including the seed
        (seed + r-1 neighbours). Corresponds to r in Algorithm 1 of Taillard &
        Voß (2018). Default 3 means seed + 2 nearest neighbours.
        max_iterations: Optional[int] = None,  # None = run until U is empty (paper default)
        base_solver: Solver for subproblems.
        base_solver_config: Configuration for base_solver.
        cluster_solver: Solver for cluster initialization.
        cluster_solver_config: Configuration for cluster initialization solver.
        initial_solver: Initial solution generation method.
        seed: Random seed.
        wastes: Node wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        vrpp: Whether to use expansion pool for VRPP.
        profit_aware_operators: Whether to use profit-aware operators.

    Returns:
        Tuple of (routes, total_cost, total_profit, info_dict).
    """
    if not mandatory:
        return [[0, 0]], 0.0, 0.0, {}

    # 1. INITIAL SOLUTION
    wastes_dict = wastes if wastes is not None else {}

    if initial_solver == "greedy":
        initial_clusters = build_greedy_routes(
            dist_matrix=distance_matrix,
            wastes=wastes_dict,
            capacity=capacity,
            R=R,
            C=C,
            mandatory_nodes=mandatory,
            rng=Random(seed),
        )
    elif initial_solver == "nearest_neighbor":
        initial_clusters = build_nn_routes(
            nodes=list(range(1, len(distance_matrix))),
            mandatory_nodes=mandatory,
            wastes=wastes_dict,
            capacity=capacity,
            dist_matrix=distance_matrix,
            R=R,
            C=C,
            rng=Random(seed),
        )
    else:
        raise ValueError(f"Unknown initial solver: {initial_solver}")

    # Build initial routes directly from clusters without running the full base solver.
    # POPMUSIC will optimise each subproblem during its iterations. Using the base
    # solver here doubles solve time for no structural benefit (paper §"Initial Solution").
    WARMSTART_INITIAL = False  # Set True only if warm-starting is intentional
    routes = []
    for cluster in initial_clusters:
        if cluster:
            if WARMSTART_INITIAL and cluster_solver is not None:
                sub_routes, _ = _optimize_subproblem(
                    base_solver=cluster_solver,
                    base_solver_config=cluster_solver_config,
                    subproblem_nodes=cluster,
                    distance_matrix=distance_matrix,
                    wastes_dict=wastes_dict,
                    capacity=capacity,
                    R=R,
                    C=C,
                    neighborhood_indices=[0],  # Dummy for initial
                    mandatory=mandatory,
                    seed=seed,
                    vrpp=vrpp,
                    profit_aware_operators=profit_aware_operators,
                )
                routes.extend(sub_routes)
            else:
                # Retain the heuristic node ordering from the initial construction directly.
                routes.append(cluster)

    # 2. POPMUSIC ITERATIONS
    iteration = 0
    # Initialize global unassigned pool for VRPP support
    unassigned_nodes: Set[int] = set()

    # Part ID management for identity tracking (Paper Algorithm 1 logic)
    # Each route is a "part" with a unique ID.
    next_part_id = len(routes)
    route_ids = list(range(len(routes)))

    # Centroid cache to avoid O(p) recomputation
    centroid_cache: Dict[int, np.ndarray] = {}

    def get_centroid(route_idx: int) -> np.ndarray:
        if route_idx not in centroid_cache:
            nodes = [n for n in routes[route_idx] if n != 0]
            if nodes:
                centroid_cache[route_idx] = coords.loc[nodes][["Lat", "Lng"]].mean().values
            else:
                centroid_cache[route_idx] = coords.loc[0][["Lat", "Lng"]].values
        return centroid_cache[route_idx]

    # Initialize Active List (U) as a stack (LIFO), queue (FIFO), or random
    if seed_strategy == "random":
        indices = list(range(len(routes)))
        Random(seed).shuffle(indices)
        active_stack = deque(indices)
    else:
        active_stack = deque(range(len(routes)))

    # Tracker for routes that have been evaluated in the current pass
    # to avoid redundant work if they were already popped.
    active_set = set(active_stack)

    while active_stack:
        if max_iterations is not None and iteration >= max_iterations:
            warnings.warn(
                f"POPMUSIC terminated early due to max_iterations={max_iterations}", RuntimeWarning, stacklevel=2
            )
            break

        # Select seed (s_g)
        i = active_stack.pop() if seed_strategy == "lifo" else active_stack.popleft()

        active_set.remove(i)

        # Skip ghost routes
        if not [n for n in routes[i] if n != 0]:
            continue

        # Recompute proximity network (KDTree) lazily if k_prox is used
        kdtree = None
        if k_prox > 0 and len(routes) > subproblem_size:
            all_centroids = np.array([get_centroid(idx) for idx in range(len(routes))])
            kdtree = KDTree(all_centroids)

        # Find R-1 nearest neighbors to route i
        neighborhood_indices = find_route_neighbors(
            i, [get_centroid(idx) for idx in range(len(routes))], subproblem_size, kdtree, k_prox
        )

        # Form subproblem
        subproblem_nodes = []
        for idx in neighborhood_indices:
            subproblem_nodes.extend([n for n in routes[idx] if n != 0])

        if not subproblem_nodes:
            continue

        # Inject nearby unassigned nodes into the subproblem (VRPP support)
        if vrpp and unassigned_nodes and subproblem_nodes:
            nearby_unassigned = []

            # Calculate average edge length in the subproblem to use as a threshold
            sub_dists = [
                distance_matrix[subproblem_nodes[x], subproblem_nodes[y]]
                for x in range(len(subproblem_nodes))
                for y in range(x + 1, len(subproblem_nodes))
            ]
            threshold = (np.mean(sub_dists) * 1.5) if sub_dists else distance_matrix.mean()

            for u_node in list(unassigned_nodes):
                min_dist_to_subproblem = min(distance_matrix[u_node, s_node] for s_node in subproblem_nodes)
                if min_dist_to_subproblem <= threshold:
                    nearby_unassigned.append(u_node)

            for node in nearby_unassigned:
                unassigned_nodes.remove(node)
                subproblem_nodes.append(node)

        # Optimize subproblem
        # get_route_cost returns raw round-trip distance (depot→nodes→depot).
        # Multiply by C to convert to cost units, matching _optimize_subproblem's profit definition.
        old_route_distance = sum(get_route_cost(distance_matrix, routes[idx]) for idx in neighborhood_indices)
        old_rev = sum(wastes_dict.get(n, 0) * R for idx in neighborhood_indices for n in routes[idx] if n != 0)
        old_profit = old_rev - old_route_distance * C

        new_routes, new_profit = _optimize_subproblem(
            base_solver=base_solver,
            base_solver_config=base_solver_config,
            subproblem_nodes=subproblem_nodes,
            distance_matrix=distance_matrix,
            wastes_dict=wastes_dict,
            capacity=capacity,
            R=R,
            C=C,
            neighborhood_indices=neighborhood_indices,
            mandatory=mandatory,
            seed=seed,
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
        )

        if new_profit > old_profit + 1e-6:
            # Update solution and manage part IDs (Paper Algorithm 1)
            modified_indices = []

            # Update existing routes
            for local_idx, global_idx in enumerate(neighborhood_indices):
                if local_idx < len(new_routes):
                    routes[global_idx] = new_routes[local_idx]
                    route_ids[global_idx] = next_part_id
                    next_part_id += 1
                    centroid_cache.pop(global_idx, None)
                    modified_indices.append(global_idx)
                else:
                    routes[global_idx] = [0, 0]
                    route_ids[global_idx] = -1  # Mark for removal
                    centroid_cache.pop(global_idx, None)
                    modified_indices.append(global_idx)

            # Handle extra routes
            if len(new_routes) > len(neighborhood_indices):
                for extra_route in new_routes[len(neighborhood_indices) :]:
                    routes.append(extra_route)
                    route_ids.append(next_part_id)
                    next_part_id += 1
                    modified_indices.append(len(routes) - 1)

            # Assert that no mutated index retains a stale centroid cache entry.
            # Development-only guard.
            for mod_idx in modified_indices:
                assert mod_idx not in centroid_cache, f"Centroid cache not cleared for modified route {mod_idx}"

            # Track unassigned nodes for VRPP support
            if vrpp:
                nodes_in_new_routes = {n for r in new_routes for n in r if n != 0}
                newly_unassigned = set(subproblem_nodes) - nodes_in_new_routes
                unassigned_nodes.update(newly_unassigned)

            # Re-insert only the parts of the optimised subproblem R into U (Paper Algorithm 1).
            # Do NOT propagate to neighbours — the paper only re-inserts direct participants.
            for mod_idx in modified_indices:
                if mod_idx >= 0 and mod_idx not in active_set and route_ids[mod_idx] != -1:
                    if seed_strategy == "lifo":
                        active_stack.append(mod_idx)
                    else:
                        active_stack.appendleft(mod_idx)
                    active_set.add(mod_idx)
        else:
            # No improvement: seed remains inactive (already popped)
            pass

        iteration += 1

        # Periodically cleanup ghost routes to prevent index-bloat
        if iteration % 50 == 0:
            valid_indices = [
                idx for idx, rid in enumerate(route_ids) if rid != -1 and [n for n in routes[idx] if n != 0]
            ]
            if len(valid_indices) < len(routes):
                # Snapshot active IDs BEFORE overwriting route_ids (fix: was computed after)
                old_active_ids = {route_ids[idx] for idx in active_set if idx < len(route_ids)}

                routes = [routes[idx] for idx in valid_indices]
                route_ids = [route_ids[idx] for idx in valid_indices]
                centroid_cache = {
                    new_idx: centroid_cache[old_idx]
                    for new_idx, old_idx in enumerate(valid_indices)
                    if old_idx in centroid_cache
                }

                # Rebuild stack from the correctly snapshotted pre-compaction IDs
                active_stack = deque([idx for idx, rid in enumerate(route_ids) if rid in old_active_ids])
                active_set = set(active_stack)

    # Calculate final summary metrics across all routes
    total_routing_cost = sum(get_route_cost(distance_matrix, r) for r in routes) * C
    total_rev = sum(wastes_dict.get(n, 0.0) * R for r in routes for n in r if n != 0)
    total_profit = total_rev - total_routing_cost

    return routes, total_routing_cost, total_profit, {"iterations": iteration, "num_routes": len(routes)}


def _optimize_subproblem(
    base_solver: Optional[str],
    base_solver_config: Optional[Any],
    subproblem_nodes: List[int],
    distance_matrix: np.ndarray,
    wastes_dict: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    neighborhood_indices: List[int],
    mandatory: List[int],
    seed: int,
    vrpp: bool = True,
    profit_aware_operators: bool = False,
) -> Tuple[List[List[int]], float]:
    """Optimize subproblem using the selected base_solver."""
    time_limit = 1.0

    # Extract the relevant config if it's nested (POPMUSICSubSolverConfig or dict)
    actual_config = base_solver_config
    if base_solver and base_solver_config:
        if hasattr(base_solver_config, base_solver):
            actual_config = getattr(base_solver_config, base_solver)
        elif isinstance(base_solver_config, dict) and base_solver in base_solver_config:
            actual_config = base_solver_config[base_solver]
            # Handle list-of-dicts style from Hydra configuration if necessary
            if isinstance(actual_config, list) and len(actual_config) > 0 and isinstance(actual_config[0], dict):
                actual_config = actual_config[0]

    if base_solver == "fast_tsp" or base_solver is None:
        return _optimize_with_fast_tsp(
            subproblem_nodes, distance_matrix, wastes_dict, capacity, R, C, actual_config, time_limit, seed, vrpp
        )
    elif base_solver == "hgs":
        return _optimize_with_hgs(
            distance_matrix,
            wastes_dict,
            capacity,
            R,
            C,
            neighborhood_indices,
            mandatory,
            actual_config,
            time_limit,
            seed,
            vrpp,
            profit_aware_operators,
            subproblem_nodes,
        )
    elif base_solver == "alns":
        return _optimize_with_alns(
            distance_matrix,
            wastes_dict,
            capacity,
            R,
            C,
            mandatory,
            actual_config,
            time_limit,
            seed,
            vrpp,
            profit_aware_operators,
            subproblem_nodes,
        )
    else:
        raise ValueError(f"Unsupported base_solver: {base_solver}")


def _optimize_with_fast_tsp(
    subproblem_nodes: List[int],
    distance_matrix: np.ndarray,
    wastes_dict: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    config: Optional[Any],
    time_limit: float,
    seed: int,
    vrpp: bool = False,
) -> Tuple[List[List[int]], float]:
    """Optimize subproblem using FastTSP and greedy split."""
    actual_time_limit = time_limit
    if config and hasattr(config, "time_limit"):
        actual_time_limit = config.time_limit
    elif isinstance(config, dict):
        actual_time_limit = config.get("time_limit", actual_time_limit)

    new_tour = find_route(distance_matrix, subproblem_nodes, time_limit=actual_time_limit, seed=seed)
    # find_route returns [0, n1, n2, ..., 0]. LinearSplit expects the giant tour EXCLUDING depot.
    giant_tour = [n for n in new_tour if n != 0]

    # Re-split using LinearSplit (optimal capacity-aware splitting)
    splitter = LinearSplit(
        dist_matrix=distance_matrix,
        wastes=wastes_dict,
        capacity=capacity,
        R=R,
        C=C,
        vrpp=vrpp,
    )
    new_routes, new_profit = splitter.split(giant_tour)
    return new_routes, new_profit


def _optimize_with_hgs(
    distance_matrix: np.ndarray,
    wastes_dict: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    neighborhood_indices: List[int],
    mandatory: List[int],
    config: Optional[Any],
    time_limit: float,
    seed: int,
    vrpp: bool = True,
    profit_aware_operators: bool = False,
    subproblem_nodes: Optional[List[int]] = None,
) -> Tuple[List[List[int]], float]:
    """Optimize subproblem using Hybrid Genetic Search (HGS)."""
    # Ensure subproblem isolation: only optimize nodes in the subproblem
    if subproblem_nodes is None:
        raise ValueError("subproblem_nodes must be provided for HGS optimization")

    # Create local mandatory: intersection of global mandatory and subproblem_nodes
    local_mandatory = [n for n in mandatory if n in subproblem_nodes]

    # Build local distance matrix: depot (0) + subproblem_nodes
    local_nodes = [0] + subproblem_nodes
    node_to_local_idx = {node: i for i, node in enumerate(local_nodes)}

    local_dist_matrix = distance_matrix[np.ix_(local_nodes, local_nodes)]
    local_wastes = {node_to_local_idx[n]: wastes_dict.get(n, 0.0) for n in subproblem_nodes}
    local_mandatory_indices = [node_to_local_idx[n] for n in local_mandatory]

    if isinstance(config, HGSConfig):
        params = HGSParams.from_config(config)
    elif isinstance(config, dict):
        # Filter for only valid HGSParams fields to avoid errors
        valid_keys = set(inspect.signature(HGSParams).parameters.keys())
        params_dict = {k: v for k, v in config.items() if k in valid_keys}
        params = HGSParams(**params_dict)
    else:
        params = HGSParams(
            max_vehicles=len(neighborhood_indices),
            time_limit=time_limit,
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
            seed=seed,
        )

    # Ensure max_vehicles and time_limit are set correctly if not in config
    if params.max_vehicles <= 0:
        params.max_vehicles = len(neighborhood_indices)
    if params.time_limit < 0:
        params.time_limit = time_limit

    solver = HGSSolver(
        dist_matrix=local_dist_matrix,
        wastes=local_wastes,
        capacity=capacity,
        R=R,
        C=C,
        params=params,
        mandatory_nodes=local_mandatory_indices,
    )
    local_routes, profit, _ = solver.solve()

    # Map local routes back to global node IDs
    routes = []
    for local_route in local_routes:
        global_route = [local_nodes[i] if i < len(local_nodes) else 0 for i in local_route]
        routes.append(global_route)

    return routes, profit


def _optimize_with_alns(
    distance_matrix: np.ndarray,
    wastes_dict: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory: List[int],
    config: Optional[Any],
    time_limit: float,
    seed: int,
    vrpp: bool = True,
    profit_aware_operators: bool = False,
    subproblem_nodes: Optional[List[int]] = None,
) -> Tuple[List[List[int]], float]:
    """Optimize subproblem using Adaptive Large Neighborhood Search (ALNS)."""
    # Ensure subproblem isolation: only optimize nodes in the subproblem
    if subproblem_nodes is None:
        raise ValueError("subproblem_nodes must be provided for ALNS optimization")

    # Create local mandatory: intersection of global mandatory and subproblem_nodes
    local_mandatory = [n for n in mandatory if n in subproblem_nodes]

    # Build local distance matrix: depot (0) + subproblem_nodes
    local_nodes = [0] + subproblem_nodes
    node_to_local_idx = {node: i for i, node in enumerate(local_nodes)}

    local_dist_matrix = distance_matrix[np.ix_(local_nodes, local_nodes)]
    local_wastes = {node_to_local_idx[n]: wastes_dict.get(n, 0.0) for n in subproblem_nodes}
    local_mandatory_indices = [node_to_local_idx[n] for n in local_mandatory]

    if config:
        if isinstance(config, ALNSConfig):
            params = ALNSParams.from_config(config)
        elif isinstance(config, dict):
            # Filter for only valid ALNSParams fields
            valid_keys = set(inspect.signature(ALNSParams).parameters.keys())
            params_dict = {k: v for k, v in config.items() if k in valid_keys}
            params = ALNSParams(**params_dict)
        else:
            params = ALNSParams(
                max_iterations=1000,
                time_limit=time_limit,
                vrpp=vrpp,
                profit_aware_operators=profit_aware_operators,
                seed=seed,
            )
    else:
        params = ALNSParams(
            max_iterations=1000,
            time_limit=time_limit,
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
        )

    # Ensure time_limit is set correctly if not in config
    if params.time_limit < 0:
        params.time_limit = time_limit

    solver = ALNSSolver(
        dist_matrix=local_dist_matrix,
        wastes=local_wastes,
        capacity=capacity,
        R=R,
        C=C,
        params=params,
        mandatory_nodes=local_mandatory_indices,
    )
    local_routes, profit, _ = solver.solve()

    # Map local routes back to global node IDs
    routes = []
    for local_route in local_routes:
        global_route = [local_nodes[i] if i < len(local_nodes) else 0 for i in local_route]
        routes.append(global_route)

    return routes, profit


def find_route_neighbors(
    seed_idx: int, centroids: List[np.ndarray], k: int, kdtree: Optional[KDTree] = None, k_prox: int = 0
) -> List[int]:
    """Find k nearest route indices to the seed route based on centroids."""
    if len(centroids) <= k:
        return list(range(len(centroids)))

    seed_pos = centroids[seed_idx]

    if kdtree is not None and k_prox > 0:
        # Use KD-Tree for proximity network acceleration (Paper §2.1)
        # Search for k_prox nearest, then take the k best from those (or just top k)
        # To match the paper's "r closest parts", we query for exactly k.
        _, indices = kdtree.query(seed_pos, k=min(k, len(centroids)))
        if isinstance(indices, (int, np.integer)):
            return [int(indices)]
        return [int(idx) for idx in indices]

    # brute-force O(p) fallback
    distances = []
    for i, pos in enumerate(centroids):
        dist = np.linalg.norm(seed_pos - pos)
        distances.append((i, dist))

    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]


def split_tour(tour: List[int], k: int, distance_matrix: np.ndarray) -> List[List[int]]:
    """Simple greedy split of a giant tour into k routes."""
    nodes = [n for n in tour if n != 0]
    if not nodes:
        return [[0, 0] for _ in range(k)]

    n = len(nodes)
    size = n // k
    remainder = n % k

    routes = []
    start = 0
    for i in range(k):
        batch_size = size + (1 if i < remainder else 0)
        route_nodes = nodes[start : start + batch_size]
        if route_nodes:
            # For POPMUSIC simplicity, we just use the order from the giant tour
            routes.append([0] + route_nodes + [0])
        else:
            routes.append([0, 0])
        start += batch_size
    return routes
