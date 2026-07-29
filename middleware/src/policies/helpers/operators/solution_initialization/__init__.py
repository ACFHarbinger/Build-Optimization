"""
Initialization schemes for VRPP.

Provides various constructive heuristics to build initial solutions:
- Greedy: Profit-driven insertion.
- Nearest Neighbor: Geographically compact clustering.
- Savings: Clarke-Wright savings algorithm.
- Regret: Regret-k insertion heuristic.
- GRASP: Randomized greedy adaptive search.
"""

from .grasp_si import build_grasp_routes
from .greedy_si import build_greedy_routes
from .nearest_neighbor_si import build_nn_routes
from .regret_si import build_regret_routes
from .savings_si import build_savings_routes

__all__ = [
    "build_greedy_routes",
    "build_nn_routes",
    "build_savings_routes",
    "build_regret_routes",
    "build_grasp_routes",
]
