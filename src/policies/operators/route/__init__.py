"""
Route Operators Package.

This package contains operator implementations for improving individual routes
or swapping nodes between routes to reduce total cost.

Attributes:
    move_swap_star (function): Best insertion swap between two routes.
    move_2opt_star (function): Exchange tails between two routes.
    move_2opt_intra (function): Reverse a market segment within a route.
    move_3opt_intra (function): Reconnect three segments within a route.

Example:
    >>> from logic.src.policies.operators.route import move_2opt_intra
    >>> improved = move_2opt_intra(ls, u=1, v=2, ...)
"""

from .swap_star import move_swap_star
from .three_opt_intra import move_3opt_intra
from .two_opt_intra import move_2opt_intra
from .two_opt_star import move_2opt_star

__all__ = [
    "move_swap_star",
    "move_2opt_star",
    "move_2opt_intra",
    "move_3opt_intra",
]
