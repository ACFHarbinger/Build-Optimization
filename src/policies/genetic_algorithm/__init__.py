"""
Genetic Algorithm (GA) solver for VRPP.

Exports:
    GASolver: The main solver class.
    GAParams: Configuration parameters dataclass.
"""

from .params import GAParams
from .solver import GASolver

__all__ = ["GASolver", "GAParams"]
