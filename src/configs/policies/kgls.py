"""
KGLS (Knowledge-Guided Local Search) configuration dataclasses.
"""

from dataclasses import dataclass, field
from typing import List

from .abc import ABCConfig


@dataclass
class KGLSConfig(ABCConfig):
    """Configuration for the Knowledge-Guided Local Search algorithm."""

    time_limit: float = 60.0
    num_perturbations: int = 3
    neighborhood_size: int = 20
    local_search_iterations: int = 100

    # Operators to apply during LS phase
    moves: List[str] = field(default_factory=lambda: ["relocate", "swap", "two_opt", "cross_exchange"])

    # Sequence of criteria to evaluate "badness" of an edge
    penalization_cycle: List[str] = field(default_factory=lambda: ["width", "length", "width_length"])

    seed: int = 42
    vrpp: bool = True
    profit_aware_operators: bool = False
