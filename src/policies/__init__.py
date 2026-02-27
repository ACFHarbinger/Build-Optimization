"""
Policies Package.

This package contains all routing policies (classical, heuristic, and neural)
used for solving the Waste Collection Vehicle Routing Problem (WCVRP) and
its variants.

Attributes:
    ALNSParams (class): Parameters for ALNS.
    NeuralAgent (class): Neural policy wrapper.
    run_alns (function): Runs ALNS algorithm.
    run_hgs (function): Runs HGS algorithm.
    find_routes (function): Solves CVRP using classical heuristics.
    find_route (function): Solves TSP.

Example:
    >>> from logic.src.policies import find_routes
    >>> routes = find_routes(distance_matrix, wastes, capacity)
"""

from .adapters.policy_vrpp import run_vrpp_optimizer
from .adaptive_large_neighborhood_search import (
    ALNSParams,
    run_alns,
    run_alns_ortools,
    run_alns_package,
)
from .ant_colony_optimization import run_hyper_heuristic_aco, run_k_sparse_aco
from .branch_cut_and_price import run_bcp
from .cvrp import find_routes, find_routes_ortools
from .hybrid_genetic_search import run_hgs
from .local_search.local_search_aco import ACOLocalSearch
from .local_search.local_search_base import LocalSearch
from .local_search.local_search_hgs import HGSLocalSearch
from .neural_agent import NeuralAgent
from .other import (
    MustGoSelectionFactory,
    MustGoSelectionRegistry,
    PostProcessorFactory,
    PostProcessorRegistry,
    SelectionContext,
)
from .simulated_annealing_neighborhood_search.common.routes import create_points
from .simulated_annealing_neighborhood_search.refinement.route_search import find_solutions
from .slack_induction_by_string_removal import run_sisr
from .tsp import find_route, get_route_cost

__all__ = [
    "ALNSParams",
    "PolicyFactory",
    "MustGoSelectionFactory",
    "MustGoSelectionRegistry",
    "SelectionContext",
    "NeuralAgent",
    "create_points",
    "create_policy",
    "find_route",
    "find_routes",
    "find_routes_ortools",
    "find_solutions",
    "get_route_cost",
    "run_alns",
    "run_alns_ortools",
    "run_alns_package",
    "run_hyper_heuristic_aco",
    "run_k_sparse_aco",
    "run_bcp",
    "run_hgs",
    "run_sisr",
    "run_vrpp_optimizer",
    "ACOLocalSearch",
    "HGSLocalSearch",
    "LocalSearch",
    "MustGoSelectionFactory",
    "MustGoSelectionRegistry",
    "SelectionContext",
    "PostProcessorFactory",
    "PostProcessorRegistry",
]

from .adapters.factory import (
    PolicyFactory,
)

create_policy = PolicyFactory.get_adapter
