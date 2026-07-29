"""
POPMUSIC (Partial Optimization Metaheuristic Under Special Intensification Conditions) Policy.

This matheuristic framework decomposes a large problem into overlapping subproblems
and optimizes them iteratively.
"""

from .policy_popmusic import POPMUSICPolicy
from .solver import run_popmusic

__all__ = ["POPMUSICPolicy", "run_popmusic"]
