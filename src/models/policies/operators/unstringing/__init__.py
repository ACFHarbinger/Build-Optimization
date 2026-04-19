"""
Vectorized Unstringing and Stringing (US) operators for WSmart-Route.

This module provides GPU-accelerated implementations of the sophisticated unstringing
operators family. Unstringing operators remove a node and reconnect the route with
multiple segment reversals, creating powerful k-opt moves that can escape deep local optima.

Operator Hierarchy (by complexity):
- Type I:   2 segment reversals, 4 arc deletions, 3 arc insertions
- Type II:  2 segment reversals (alternative ordering)
- Type III: 3 segment reversals, 5 arc deletions, 4 arc insertions
- Type IV:  Complex rearrangement with selective reversals (most powerful)

All operators support batch processing for parallel evaluation across multiple instances.
"""

from .type_i import vectorized_type_i_unstringing
from .type_ii import vectorized_type_ii_unstringing
from .type_iii import vectorized_type_iii_unstringing
from .type_iv import vectorized_type_iv_unstringing

__all__ = [
    "vectorized_type_i_unstringing",
    "vectorized_type_ii_unstringing",
    "vectorized_type_iii_unstringing",
    "vectorized_type_iv_unstringing",
]
