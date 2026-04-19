"""
Differential Evolution (DE) policy for VRPP.

Rigorous implementation of Storn & Price (1997) DE/rand/1/bin algorithm,
replacing the metaphor-heavy Artificial Bee Colony (ABC) implementation.
"""

from .params import DEParams
from .solver import DESolver

__all__ = ["DEParams", "DESolver"]
