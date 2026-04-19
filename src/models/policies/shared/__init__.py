"""
Split Package.

Exports:
    vectorized_linear_split
    vectorized_split_limited (_vectorized_split_limited)
    reconstruct_routes (_reconstruct_routes)
    reconstruct_limited (_reconstruct_limited)
"""

from .limited import vectorized_split_limited
from .linear import vectorized_linear_split
from .reconstruction import reconstruct_limited, reconstruct_routes

# Aliases for backward compatibility
_vectorized_split_limited = vectorized_split_limited
_reconstruct_routes = reconstruct_routes
_reconstruct_limited = reconstruct_limited

__all__ = [
    "vectorized_linear_split",
    "_vectorized_split_limited",
    "_reconstruct_routes",
    "_reconstruct_limited",
]
