"""
Exchange Operators Package.

This package contains operator implementations for exchanging segments between routes,
including Cross-Exchange, Ejection Chains, and Lambda-Interchange.

Attributes:
    move_or_opt (function): Or-opt operator for relocating chains.
    cross_exchange (function): Exchange segments between two routes.
    ejection_chain (function): Compound displacement for fleet reduction.
    lambda_interchange (function): Generalized exchange operator.

Example:
    >>> from logic.src.policies.operators.exchange import cross_exchange
    >>> improved = cross_exchange(ls, r_a=0, ...)
"""

from .cross import cross_exchange
from .ejection import ejection_chain
from .lambda_interchange import lambda_interchange
from .or_opt import move_or_opt

__all__ = [
    "move_or_opt",
    "cross_exchange",
    "ejection_chain",
    "lambda_interchange",
]
