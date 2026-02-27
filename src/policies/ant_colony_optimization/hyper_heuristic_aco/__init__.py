"""
Hyper-Heuristic Ant Colony Optimization.

This package implements a hyper-heuristic approach where ants optimize the
sequence of local search operators applied to a solution, rather than
constructing the solution itself.

Attributes:
    HYPER_OPERATORS (list): List of available hyper-heuristic operators.
    HyperACOParams (class): Parameters for the Hyper-ACO algorithm.
    HyperHeuristicACO (class): The main solver class.
    run_hyper_heuristic_aco (function): Helper function to run the solver.

Example:
    >>> from logic.src.policies.ant_colony_optimization.hyper_heuristic_aco import run_hyper_heuristic_aco
    >>> result = run_hyper_heuristic_aco(dist_matrix, wastes, ...)
"""

from .hyper_aco import HyperHeuristicACO
from .hyper_operators import HYPER_OPERATORS, OPERATOR_NAMES
from .params import HyperACOParams
from .runner import run_hyper_heuristic_aco

__all__ = ["HyperHeuristicACO", "HyperACOParams", "HYPER_OPERATORS", "OPERATOR_NAMES", "run_hyper_heuristic_aco"]
