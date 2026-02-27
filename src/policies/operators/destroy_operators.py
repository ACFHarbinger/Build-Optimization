"""
Destroy Operators Module.

This module exposes the various destroy (removal) operators used in the
Adaptive Large Neighborhood Search (ALNS) algorithm.

Attributes:
    random_removal (function): Randomly removes nodes.
    worst_removal (function): Removes nodes with highest cost contribution.
    cluster_removal (function): Removes spatially clustered nodes.
    shaw_removal (function): Removes related nodes (distance/time/waste).
    string_removal (function): Removes contiguous sequences of nodes.

Example:
    >>> from logic.src.policies.operators import destroy_operators
    >>> routes, removed = destroy_operators.random_removal(routes, n=5)
"""

from .destroy import (
    cluster_removal,
    random_removal,
    shaw_removal,
    string_removal,
    worst_removal,
)

__all__ = [
    "random_removal",
    "worst_removal",
    "cluster_removal",
    "shaw_removal",
    "string_removal",
]
