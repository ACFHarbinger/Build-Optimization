"""
Hybrid Genetic Search with Adaptive Large Neighborhood Search (HGS-ALNS).

Combines HGS evolutionary operators with ALNS-based education phase for
intensive local search optimization.
"""

from .hgs_alns import HGSALNSSolver
from .params import HGSALNSParams

__all__ = ["HGSALNSSolver", "HGSALNSParams"]
