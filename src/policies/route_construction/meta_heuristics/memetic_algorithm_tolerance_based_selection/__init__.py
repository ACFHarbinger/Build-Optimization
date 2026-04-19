"""
Genetic Algorithm with Stochastic Tournament Selection for VRPP.

Rigorous implementation replacing "League Championship Algorithm (LCA)".
"""

from .params import MemeticAlgorithmToleranceBasedSelectionParams
from .solver import MemeticAlgorithmToleranceBasedSelectionSolver

__all__ = ["MemeticAlgorithmToleranceBasedSelectionSolver", "MemeticAlgorithmToleranceBasedSelectionParams"]
