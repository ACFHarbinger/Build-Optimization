"""
Simulated Annealing (SA) solver for VRPP.

Exports:
    SASolver: The main solver class.
    SAParams: Configuration parameters dataclass.
"""

from .params import SAParams
from .solver import SASolver

__all__ = ["SASolver", "SAParams"]
