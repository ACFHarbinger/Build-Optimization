"""
Repair Operators Module.

This module exposes the repair (insertion) operators used to reconstruct
partial solutions in the ALNS algorithm.

Attributes:
    greedy_insertion (function): Cheapest insertion.
    regret_2_insertion (function): 2-regret insertion.
    regret_k_insertion (function): k-regret insertion.
    greedy_insertion_with_blinks (function): Randomized greedy insertion.

Example:
    >>> from logic.src.policies.operators import repair_operators
    >>> routes = repair_operators.greedy_insertion(routes, removed, ...)
"""

from .repair import (
    greedy_insertion,
    greedy_insertion_with_blinks,
    regret_2_insertion,
    regret_k_insertion,
)

__all__ = [
    "greedy_insertion",
    "regret_2_insertion",
    "regret_k_insertion",
    "greedy_insertion_with_blinks",
]
