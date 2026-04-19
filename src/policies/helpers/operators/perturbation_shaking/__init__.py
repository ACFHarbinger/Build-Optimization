"""
Perturbation Operators Package.

This package contains operator implementations for perturbing the current
solution to escape local optima.

Includes:
    - kick: Large-scale destroy & repair.
    - perturb: Small-scale random changes.
    - double_bridge: 4-opt double-bridge move.
    - genetic_transformation: Elite-guided edge preservation.
    - evolutionary_perturbation: Micro-GA on route clusters.
    - bb_perturbation: B&B destroy-and-repair (Shaw 1998).

Example:
    >>> from logic.src.policies.helpers.operators.perturbation import kick
    >>> kick(context, destroy_ratio=0.2)
"""

from .branch_and_bound import bb_perturbation, bb_profit_perturbation
from .cross_day import cross_day_move, cross_day_swap, day_merge_split
from .day_shuffle import day_shuffle
from .double_bridge import double_bridge
from .evolutionary import (
    evolutionary_perturbation,
    evolutionary_perturbation_profit,
)
from .genetic_transformation import (
    genetic_transformation,
    genetic_transformation_profit,
)
from .kick import kick, kick_profit
from .perturb import perturb, perturb_profit

__all__ = [
    "bb_perturbation",
    "bb_profit_perturbation",
    "perturb",
    "perturb_profit",
    "kick",
    "kick_profit",
    "double_bridge",
    "genetic_transformation",
    "genetic_transformation_profit",
    "evolutionary_perturbation",
    "evolutionary_perturbation_profit",
    "day_shuffle",
    "cross_day_move",
    "cross_day_swap",
    "day_merge_split",
]
