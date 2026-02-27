"""
Slack Induction by String Removal (SISR) Package.

This package implements the SISR heuristic for the Vehicle Routing Problem.
SISR is an iterated local search method that destroys routes by removing strings
of nodes and repairs them using a greedy heuristic with blinks.

Attributes:
    SISRParams (class): Configuration parameters.
    SISRSolver (class): Main solver class.
    run_sisr (function): Helper function to run the solver.

Example:
    >>> from logic.src.policies.slack_induction_by_string_removal import run_sisr
    >>> result = run_sisr(dist_matrix, wastes, ...)
"""

from .sisr import SISRParams, SISRSolver, run_sisr
