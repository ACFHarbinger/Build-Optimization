"""
Memetic Algorithm with Dual Population (MA-DP) for VRPP.

Rigorous implementation based on VPL but with OR terminology.
Maintains dual population structure (active + reserve).
"""

from .params import MemeticAlgorithmDualPopulationParams
from .policy_ma_dp import MemeticAlgorithmDualPopulationPolicy
from .solver import MemeticAlgorithmDualPopulationSolver

__all__ = [
    "MemeticAlgorithmDualPopulationSolver",
    "MemeticAlgorithmDualPopulationParams",
    "MemeticAlgorithmDualPopulationPolicy",
]
