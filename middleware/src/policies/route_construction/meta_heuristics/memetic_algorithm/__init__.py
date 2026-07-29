"""
Memetic Algorithm (MA) implementation for VRPP.

Follows the Moscato, Cotta, & Mendes (2004) framework:
- Population-based evolutionary search (GA)
- Local Search refinement (Hill-climbing)
"""

from .params import MAParams
from .policy_ma import MAPolicy
from .solver import MASolver

__all__ = [
    "MASolver",
    "MAParams",
    "MAPolicy",
]
