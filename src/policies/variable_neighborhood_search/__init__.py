"""
Variable Neighborhood Search (VNS) solver for VRPP.

Exports:
    VNSSolver: The main solver class.
    VNSParams: Configuration parameters dataclass.
"""

from .params import VNSParams
from .solver import VNSSolver

__all__ = ["VNSSolver", "VNSParams"]
