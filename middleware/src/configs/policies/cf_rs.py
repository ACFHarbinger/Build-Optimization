"""
Cluster-First Route-Second (CF-RS) configuration schemas.

This module defines the Hydra-compatible configuration dataclasses for the CF-RS
routing policy, following the project's standard configuration architecture.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class CFRSConfig:
    """Configuration for Cluster-First Route-Second (CF-RS) policy.

    Based on the geometric decomposition heuristic proposed by Fisher and Jaikumar (1981):
    "A generalized assignment heuristic for vehicle routing".

    This implementation focuses on angular partitioning of the customer set
    relative to a central depot, which is highly effective for radially
    distributed service areas.

    Attributes:
        vrpp: Whether the problem is a VRP with Profits.
        seed: Random seed for reproducibility in the TSP routing phase.
            Ensures that the stochastic elements of the TSP solver (e.g., 2-opt
            initialization) yield consistent results.
        num_clusters: Number of angular sectors to partition nodes into.
            If set to 0, the policy defaults to using the number of vehicles
            available in the simulation day (`n_vehicles`).
        time_limit: Maximum time in seconds allowed for the routing phase
            (total across all sectors).
        assignment_method: Method for assigning nodes to clusters.
            Options: "greedy" (Fisher & Jaikumar heuristic, default) or
            "exact" (Mixed-Integer Programming via Gurobi).
        route_optimizer: TSP solver for routing phase.
            Options: "default" (fast_tsp), "pso" (Particle Swarm Optimization),
            "aco" (Ant Colony Optimization), "ga" (Genetic Algorithm).
            VFJ paper (Sultana & Akhand, 2017) recommends "pso" for best performance.
        strict_fleet: Enforce fixed fleet size K (benchmark mode).
            If True, raises ValueError if greedy assignment cannot fit all nodes.
            If False, dynamically opens new vehicles as needed (simulation mode).
        seed_criterion: Seed selection method ("distance" or "demand").
            "distance" selects furthest node from depot in each sector (default).
            "demand" selects node with maximum waste in each sector.
            VFJ paper Section 3.2: Two methods for seed selection.
        mip_objective: MIP objective for exact assignment mode.
            "minimize_cost" (default): Minimize total insertion cost for benchmarks.
            "maximize_profit": Maximize profit (revenue - cost) for simulation.
        mandatory_selection: List of mandatory strategy configuration files or dicts.
            Controls which bins are selected for collection on a given day.
        route_improvement: List of route improvement operations (e.g., local search)
            to apply to the resulting tours for further refinement.
    """

    vrpp: bool = True

    # Reproducibility seed for Routing phase
    seed: Optional[int] = None

    # Clustering granularity
    num_clusters: int = 0

    # Routing time limit (seconds)
    time_limit: float = 60.0

    # Assignment method: "greedy" or "exact"
    assignment_method: str = "greedy"

    # TSP routing optimizer: "default", "pso", "aco", "ga"
    # VFJ paper recommends "pso" for best performance
    route_optimizer: str = "default"

    # Strict fleet sizing for benchmark compliance
    # If True, raises error if K vehicles insufficient (standard CVRP mode)
    # If False, dynamically opens new vehicles (simulation mode)
    strict_fleet: bool = False

    # Seed selection criterion: "distance" or "demand"
    # VFJ paper Section 3.2: "a. The most distant node from the origin and
    # b. The node with maximum demand."
    seed_criterion: str = "distance"

    # MIP objective function: "minimize_cost" or "maximize_profit"
    # Use "minimize_cost" for benchmark compliance with A-VRP dataset
    mip_objective: str = "minimize_cost"

    # Bin selection strategies (VRPP/WCVRP specific)
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None

    # Tour refinement strategies
    route_improvement: Optional[List[RouteImprovingConfig]] = None
