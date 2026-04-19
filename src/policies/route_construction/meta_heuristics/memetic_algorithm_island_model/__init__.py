"""
Memetic Island Model Genetic Algorithm for VRPP.

Rigorous implementation based on SLC but with GA terminology.
Uses an island model architecture with memetic local search.
"""

from .params import MemeticAlgorithmIslandModelParams
from .solver import MemeticAlgorithmIslandModelSolver

__all__ = ["MemeticAlgorithmIslandModelSolver", "MemeticAlgorithmIslandModelParams"]
