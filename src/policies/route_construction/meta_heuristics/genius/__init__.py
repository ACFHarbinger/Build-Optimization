"""
GENIUS (GENI + US) Meta-Heuristic for VRPP.

Implements the algorithm from Gendreau, Hertz, and Laporte (1992) combining:
- GENI (Generalized Insertion) for construction
- US (Unstringing and Stringing) for post-optimization

This implementation provides:
1. Deterministic p-neighborhood search (strict adherence to GHL 1992)
2. Randomized sampling mode (configurable via random_us_sampling flag)
3. Corrected US acceptance criterion (accepts all moves, tracks global best)

Reference:
    Gendreau, M., Hertz, A., & Laporte, G. (1992).
    "New Insertion and Postoptimization Procedures for the Traveling Salesman Problem"
    Operations Research, 40(6), 1086-1094.

Exports:
    GENIUSSolver: The main solver class.
    GENIUSParams: Configuration parameters dataclass.
    GENIUSPolicy: Policy adapter for the routing framework.
"""

from .params import GENIUSParams
from .policy_genius import GENIUSPolicy
from .solver import GENIUSSolver

__all__ = ["GENIUSSolver", "GENIUSParams", "GENIUSPolicy"]
