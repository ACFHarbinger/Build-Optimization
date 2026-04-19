"""
Core implementation of Cluster-First Route-Second (CF-RS) Algorithm.

This module provides the computational logic for the two-stage routing heuristic:
1. Clustering: Partitioning nodes into angular sectors centered at the depot.
2. Routing: Solving a Traveling Salesman Problem (TSP) for each sector independently.

This approach is particularly efficient for large-scale logistics where global
optimization is computationally prohibitive, and provides intuitive,
geometrically-grouped routes.

The module supports two assignment methods:
- **Greedy**: Original Fisher & Jaikumar heuristic (Sultana & Akhand, 2017)
- **Exact**: Mixed-Integer Programming using Gurobi (maximizes profit)

Reference:
    Fisher, M. L., & Jaikumar, R. (1981). "A generalized assignment heuristic
    for vehicle routing". Networks, 11(2), 109-124.
    Sultana, T., Akhand, M. A. H., & Rahman, M. M. H. (2017). "A Variant Fisher
    and Jaikumar Algorithm to Solve Capacitated Vehicle Routing Problem".
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from logic.src.policies.route_construction.matheuristics.cluster_first_route_second.greedy_assignment import (
    assign_greedy,
)
from logic.src.policies.route_construction.matheuristics.cluster_first_route_second.mip_assignment import (
    assign_exact_mip,
)
from logic.src.policies.route_construction.matheuristics.cluster_first_route_second.tsp_metaheuristics import (
    find_route_aco,
    find_route_ga,
    find_route_pso,
)
from logic.src.policies.route_construction.other_algorithms.travelling_salesman_problem.tsp import (
    calculate_tour_cost,
    find_route,
)


def run_cf_rs(
    coords: pd.DataFrame,
    mandatory: List[int],
    distance_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    n_vehicles: int,
    seed: int = 42,
    num_clusters: int = 0,
    time_limit: float = 60.0,
    assignment_method: str = "greedy",
    route_optimizer: str = "default",
    strict_fleet: bool = False,
    seed_criterion: str = "distance",
    mip_objective: str = "minimize_cost",
) -> Tuple[List[List[int]], float, Dict[str, Any]]:
    """
    Solve VRPP using Cluster-First Route-Second angular partitioning (VFJ Algorithm).

    This function orchestrates the two-stage decomposition of the routing problem:
    1. Clustering: VFJ partitioning (equal nodes per sector) + assignment
    2. Routing: TSP solving with configurable metaheuristics

    VFJ Algorithm Enhancements (Sultana & Akhand, 2017):
    - Equal-node-count partitioning (vs. equal-angle in SFJ 1981)
    - Support for metaheuristic TSP solvers (PSO, ACO, GA)
    - Strict fleet sizing for benchmark compliance

    Args:
        coords: DataFrame containing 'Lat' and 'Lng' for all nodes.
            The depot must be at index 0.
        mandatory: List of global indices of bins to be collected.
        distance_matrix: Pre-computed all-pairs distance matrix.
        wastes: Dictionary mapping node indices to waste quantities.
        capacity: Maximum vehicle capacity.
        R: Revenue per unit of waste.
        C: Cost per unit of distance.
        n_vehicles: Total number of vehicles available for the simulation day.
        seed: Random seed for the TSP solver to ensure deterministic results.
        num_clusters: Explicit number of sectors to create. If 0, uses n_vehicles.
        time_limit: Maximum time in seconds for optimization.
        assignment_method: Assignment strategy ("greedy" or "exact").
        route_optimizer: TSP solver ("default", "pso", "aco", "ga").
            Paper recommends "pso" for best performance.
        strict_fleet: If True, enforce fixed fleet size K (benchmark mode).
        seed_criterion: Seed selection method ("distance" or "demand").
            "distance" selects furthest node from depot in each sector.
            "demand" selects node with maximum waste in each sector.
        mip_objective: MIP objective for exact assignment ("minimize_cost" or "maximize_profit").
            Use "minimize_cost" for benchmark compliance with A-VRP dataset.

    Returns:
        A tuple containing:
            - List[List[int]]: Routes for each vehicle.
            - float: Total distance of the concatenated tours.
            - Dict[str, Any]: Metadata containing the cluster assignments for analysis.

    Raises:
        ValueError: If route_optimizer is not recognized or strict_fleet fails.

    Reference:
        Sultana, T., Akhand, M. A. H., & Rahman, M. M. H. (2017).
        "A Variant Fisher and Jaikumar Algorithm to Solve Capacitated Vehicle Routing Problem"
    """
    if not mandatory:
        return [[0, 0]], 0.0, {}

    # Determine number of clusters (K)
    if num_clusters > 0:
        k = num_clusters
    else:
        # Dynamic calculation: as many clusters as necessary to fit total waste
        total_waste = sum(wastes.get(node, 0) for node in mandatory)
        # At least 1 cluster if nodes exist, and at least n_vehicles
        min_k = math.ceil(total_waste / capacity) if capacity > 0 else 1
        k = max(n_vehicles, min_k)
        if k == 0 and mandatory:
            k = 1

    # STAGE 1: CLUSTERING (VFJ Methodology)
    # VFJ partitions by equal node count per sector (key contribution vs. SFJ 1981)
    clusters = fisher_jaikumar_clustering(
        coords,
        mandatory,
        k,
        wastes,
        capacity,
        R,
        C,
        distance_matrix,
        time_limit,
        assignment_method,
        strict_fleet,
        seed_criterion,
        mip_objective,
    )

    # STAGE 2: ROUTING (Metaheuristic TSP Solvers)
    # VFJ paper evaluates PSO, ACO, and GA for TSP phase
    # Conclusion: VFJ + PSO provides best performance
    routes = []
    total_cluster_cost = 0.0

    # Select TSP solver based on route_optimizer parameter
    if route_optimizer == "pso":
        tsp_solver = find_route_pso
    elif route_optimizer == "aco":
        tsp_solver = find_route_aco  # type: ignore[assignment]
    elif route_optimizer == "ga":
        tsp_solver = find_route_ga  # type: ignore[assignment]
    elif route_optimizer == "default":
        tsp_solver = find_route  # type: ignore[assignment]
    else:
        raise ValueError(
            f"Invalid route_optimizer: '{route_optimizer}'. "
            f"Must be one of: 'default', 'pso', 'aco', 'ga'. "
            f"Paper recommends 'pso' for best performance."
        )

    for cluster in clusters:
        if not cluster:
            continue

        # Solve TSP for the current sector using selected algorithm
        cluster_tour = tsp_solver(distance_matrix, cluster, time_limit=time_limit, seed=seed)

        # Remove the leading/trailing 0s if find_route includes them,
        # or keep the structure expected by the framework.
        clean_route = [node for node in cluster_tour if node != 0]
        if clean_route:
            routes.append(clean_route)

        cluster_cost = calculate_tour_cost(distance_matrix, cluster_tour)
        total_cluster_cost += cluster_cost

    return routes, total_cluster_cost, {"clusters": clusters, "num_sectors": len(clusters)}


def fisher_jaikumar_clustering(
    coords: pd.DataFrame,
    mandatory: List[int],
    k: int,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    distance_matrix: np.ndarray,
    time_limit: float = 60.0,
    assignment_method: str = "greedy",
    strict_fleet: bool = True,
    seed_criterion: str = "distance",
    mip_objective: str = "minimize_cost",
) -> List[List[int]]:
    """
    Partition bins using the VFJ (Variant Fisher & Jaikumar) clustering algorithm.

    VFJ Algorithm (Sultana & Akhand, 2017):
    1. Select K seed nodes using equal-node-count partitioning (NOT equal-angle).
       This is the key difference from the original SFJ algorithm (1981).
    2. Assign each node to the seed that minimizes insertion cost:
       c_ik = d(0, i) + d(i, k) - d(0, k)
    3. Respect capacity constraints.

    Args:
        coords: DataFrame containing 'Lat' and 'Lng' for all nodes.
        mandatory: List of global indices of bins to be collected.
        k: Number of clusters/seeds to create.
        wastes: Dictionary mapping node indices to waste quantities.
        capacity: Maximum vehicle capacity.
        R: Revenue per unit of waste.
        C: Cost per unit of distance.
        distance_matrix: Pre-computed all-pairs distance matrix.
        time_limit: Maximum time in seconds for optimization.
        assignment_method: Assignment strategy ("greedy" or "exact").
        strict_fleet: If True, enforce fixed fleet size K (benchmark mode).
        seed_criterion: Seed selection method ("distance" or "demand").
            Paper Section 3.2: "a. The most distant node from the origin and
            b. The node with maximum demand."
        mip_objective: MIP objective for exact assignment ("minimize_cost" or "maximize_profit").
            Use "minimize_cost" for benchmark compliance with A-VRP dataset.

    Returns:
        List of clusters, where each cluster is a list of node indices.

    Raises:
        ValueError: If assignment_method is not "greedy" or "exact",
            if strict_fleet=True and greedy assignment fails,
            if seed_criterion is invalid, or if mip_objective is invalid.

    Reference:
        Sultana, T., Akhand, M. A. H., & Rahman, M. M. H. (2017).
        Section 3.2: "each cone contains equal number of nodes"
    """
    if not mandatory:
        return []

    # Validate assignment method
    if assignment_method not in ["greedy", "exact"]:
        raise ValueError(f"Invalid assignment_method: '{assignment_method}'. Must be 'greedy' or 'exact'.")

    # 1. Depot and node feature extraction
    depot_lat, depot_lng = _get_depot_coords(coords)
    df_nodes = _compute_node_features(coords, mandatory, depot_lat, depot_lng, distance_matrix)

    # 2. Seed Selection (VFJ: equal node count per sector)
    seeds = _select_initial_seeds(df_nodes, k, wastes, seed_criterion)
    if not seeds:
        return []

    # 3. Assignment - Route to appropriate method
    if assignment_method == "greedy":
        clusters = assign_greedy(seeds, mandatory, wastes, capacity, distance_matrix, strict_fleet)
    else:  # assignment_method == "exact"
        clusters = assign_exact_mip(
            seeds, mandatory, wastes, capacity, R, C, distance_matrix, time_limit, mip_objective
        )  # type: ignore[assignment]
        # If MIP assignment fails or Gurobi unavailable, fall back to greedy
        if clusters is None:
            clusters = assign_greedy(seeds, mandatory, wastes, capacity, distance_matrix, strict_fleet)

    # Filter out empty clusters
    return [c for c in clusters if c]


def _get_depot_coords(coords: Any) -> Tuple[float, float]:
    """Extract coordinates of the depot node (index 0)."""
    if isinstance(coords, pd.DataFrame):
        return float(coords.iloc[0]["Lat"]), float(coords.iloc[0]["Lng"])
    return float(coords[0, 0]), float(coords[0, 1])


def _compute_node_features(
    coords: Any,
    mandatory: List[int],
    depot_lat: float,
    depot_lng: float,
    distance_matrix: np.ndarray,
) -> pd.DataFrame:
    """Compute angular positions and distances for all nodes relative to depot."""
    node_data = []
    for idx in mandatory:
        if isinstance(coords, pd.DataFrame):
            lat, lng = coords.iloc[idx]["Lat"], coords.iloc[idx]["Lng"]
        else:
            lat, lng = coords[idx, 0], coords[idx, 1]
        angle = math.atan2(lat - depot_lat, lng - depot_lng)
        dist = distance_matrix[0, idx]
        node_data.append({"idx": idx, "angle": angle, "dist": dist})

    return pd.DataFrame(node_data).sort_values("angle")


def _select_initial_seeds(
    df_nodes: pd.DataFrame, k: int, wastes: Optional[Dict[int, float]] = None, seed_criterion: str = "distance"
) -> List[int]:
    """
    Partition space into K sectors and pick seed node in each based on criterion.

    VFJ Methodology (Sultana & Akhand, 2017):
    Unlike the original Fisher & Jaikumar (1981) equal-angle partitioning,
    VFJ partitions by EQUAL NUMBER OF NODES in each angular sector.
    This is the key algorithmic contribution of the VFJ variant.

    Seed Selection Criteria (Section 3.2):
    The paper states: "Seed customer from each cone can be selected in one of
    the following two ways: a. The most distant node from the origin and
    b. The node with maximum demand."

    Algorithm:
    1. Sort all nodes by angular coordinate relative to depot
    2. Divide sorted list into K contiguous chunks (equal node count)
    3. Select the seed in each chunk based on criterion:
       - "distance": furthest node from depot
       - "demand": node with maximum waste/demand

    Args:
        df_nodes: DataFrame with columns ['idx', 'angle', 'dist']
        k: Number of clusters/seeds to create
        wastes: Dictionary mapping node indices to waste quantities (required for "demand" criterion)
        seed_criterion: Selection criterion ("distance" or "demand")

    Returns:
        List of seed node indices

    Raises:
        ValueError: If seed_criterion is not "distance" or "demand",
            or if wastes is None when seed_criterion is "demand".

    Reference:
        Sultana, T., Akhand, M. A. H., & Rahman, M. M. H. (2017).
        Section 3.2: "each cone contains equal number of nodes"
    """
    # Validate seed criterion
    if seed_criterion not in ["distance", "demand"]:
        raise ValueError(
            f"Invalid seed_criterion: '{seed_criterion}'. "
            f"Must be 'distance' (furthest from depot) or 'demand' (maximum waste)."
        )

    # Validate wastes parameter for demand criterion
    if seed_criterion == "demand" and wastes is None:
        raise ValueError(
            "wastes parameter is required when seed_criterion='demand'. "
            "Please provide a dictionary mapping node indices to waste quantities."
        )

    seeds: List[int] = []

    # Sort by angular coordinate (counter-clockwise from depot)
    df_sorted = df_nodes.sort_values("angle").reset_index(drop=True)
    n_nodes = len(df_sorted)

    if n_nodes == 0:
        return seeds

    # Calculate nodes per sector (distribute remainder evenly)
    nodes_per_sector = n_nodes // k
    remainder = n_nodes % k

    start_idx = 0
    for i in range(k):
        # Distribute remainder nodes to first few sectors
        sector_size = nodes_per_sector + (1 if i < remainder else 0)

        if sector_size == 0:
            break

        end_idx = start_idx + sector_size
        sector = df_sorted.iloc[start_idx:end_idx]

        if not sector.empty:
            if seed_criterion == "distance":
                # VFJ Method a: Select the furthest node from depot
                seed_idx = int(sector.loc[sector["dist"].idxmax(), "idx"])
            else:  # seed_criterion == "demand"
                # VFJ Method b: Select the node with maximum demand (waste)
                # Add waste column to sector for selection
                sector_with_waste = sector.copy()
                sector_with_waste["waste"] = sector_with_waste["idx"].map(lambda x: wastes.get(x, 0.0))  # type: ignore[union-attr]
                seed_idx = int(sector_with_waste.loc[sector_with_waste["waste"].idxmax(), "idx"])

            seeds.append(seed_idx)

        start_idx = end_idx

    return seeds
