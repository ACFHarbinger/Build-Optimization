"""
Perturbation Operators Module.

This module exposes the perturbation operators used to escape local optima
in the search process.

Attributes:
    kick (function): Destroys and repairs part of the solution.
    perturb (function): Performs random swaps.

Example:
    >>> from logic.src.policies.operators import perturbation_operators
    >>> perturbation_operators.kick(context, destroy_ratio=0.3)
"""

from .perturbation import (
    kick,
    perturb,
)

__all__ = [
    "perturb",
    "kick",
]
