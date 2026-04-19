"""
Cluster-First Route-Second (CF-RS) Routing Policy.

This package implements the CF-RS algorithm, a variant of the Fisher and Jaikumar (1981)
heuristic. It partitions nodes into angular sectors centered at the depot (Clustering)
and solves the Traveling Salesman Problem (TSP) for each sector (Routing).

The package supports two assignment methods:
- Greedy: Original Fisher & Jaikumar heuristic (Sultana & Akhand, 2017)
- Exact: Mixed-Integer Programming using Gurobi

Modules:
    solver: Core implementation of the angular clustering and TSP routing.
    greedy_assignment: Greedy heuristic for node-to-cluster assignment.
    mip_assignment: Exact MIP formulation for node-to-cluster assignment.
    policy_cf_rs: Simulator adapter for the CF-RS algorithm.
"""

from .greedy_assignment import assign_greedy
from .mip_assignment import assign_exact_mip
from .policy_cf_rs import ClusterFirstRouteSecondPolicy
from .solver import run_cf_rs
from .tsp_metaheuristics import find_route_aco, find_route_ga, find_route_pso

__all__ = [
    "ClusterFirstRouteSecondPolicy",
    "run_cf_rs",
    "assign_greedy",
    "assign_exact_mip",
    "find_route_pso",
    "find_route_aco",
    "find_route_ga",
]
