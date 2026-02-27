"""
Repair Operators Package.

This package contains operator implementations for repairing (re-inserting nodes into)
partial solutions. Includes greedy and regret-based heuristics.

Attributes:
    greedy_insertion (function): Best single-move insertion.
    greedy_insertion_with_blinks (function): Randomized greedy insertion.
    regret_2_insertion (function): Insertion maximizing difference between best and 2nd best.
    regret_k_insertion (function): General k-regret insertion.

Example:
    >>> from logic.src.policies.operators.repair import greedy_insertion
    >>> new_routes = greedy_insertion(routes, removed, dist_matrix, wastes, capacity)
"""

from .greedy import greedy_insertion
from .greedy_blink import greedy_insertion_with_blinks
from .regret import regret_2_insertion, regret_k_insertion

__all__ = [
    "greedy_insertion",
    "regret_2_insertion",
    "regret_k_insertion",
    "greedy_insertion_with_blinks",
]
