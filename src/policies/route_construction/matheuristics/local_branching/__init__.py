"""
Local Branching (LB) matheuristic package.

Algorithm:
    Introduced by Matteo Fischetti and Andrea Lodi (2003), Local Branching
    defines a systematic neighborhood search in Mixed-Integer Programming
    by leveraging standard MIP solvers.

Key Feature:
    The use of linear 'Hamming distance' constraints to define and explore
    regions near high-quality known solutions without having to solve the
    global problem to optimality in one step.
"""

from .policy_lb import LocalBranchingPolicy

__all__ = ["LocalBranchingPolicy"]
