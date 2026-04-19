"""
Slack Induction by String Removal (SISR) Package.

This package implements the SISR heuristic for the Vehicle Routing Problem.
SISR is an iterated local search method that destroys routes by removing strings
of nodes and repairs them using a greedy heuristic with blinks.

Attributes:
    SISRParams (class): Configuration parameters.
    SISRSolver (class): Main solver class.
"""

from .params import SISRParams
from .solver import SISRSolver
