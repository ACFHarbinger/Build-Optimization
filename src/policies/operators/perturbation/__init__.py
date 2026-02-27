"""
Perturbation Operators Package.

This package contains operator implementations for perturbing the current
solution to escape local optima.

Attributes:
    kick (function): Large-scale neighborhood change (destroy & repair).
    perturb (function): Small-scale random changes.

Example:
    >>> from logic.src.policies.operators.perturbation import kick
    >>> kick(context, destroy_ratio=0.2)
"""

from .kick import kick
from .perturb import perturb

__all__ = [
    "perturb",
    "kick",
]
