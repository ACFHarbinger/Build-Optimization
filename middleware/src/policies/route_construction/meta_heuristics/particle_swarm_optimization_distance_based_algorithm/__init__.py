"""
Distance-Based Particle Swarm Optimization for VRPP.

Rigorous implementation replacing metaphor-based "Firefly Algorithm".
"""

from .params import DistancePSOParams
from .solver import DistancePSOSolver

__all__ = ["DistancePSOSolver", "DistancePSOParams"]
