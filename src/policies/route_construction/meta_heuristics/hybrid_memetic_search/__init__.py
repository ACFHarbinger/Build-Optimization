"""
Hybrid Memetic Search (HMS) for VRPP.

Rigorous implementation replacing "Hybrid Volleyball Premier League (HVPL)".
Multi-phase hybrid solver combining ACO, GA, and ALNS.
"""

from .params import HybridMemeticSearchParams
from .solver import HybridMemeticSearchSolver

__all__ = ["HybridMemeticSearchSolver", "HybridMemeticSearchParams"]
