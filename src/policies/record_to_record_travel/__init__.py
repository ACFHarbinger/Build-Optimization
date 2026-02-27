"""
Record-to-Record Travel (RR) solver for VRPP.

Exports:
    RRSolver: The main solver class.
    RRParams: Configuration parameters dataclass.
"""

from .params import RRParams
from .solver import RRSolver

__all__ = ["RRSolver", "RRParams"]
