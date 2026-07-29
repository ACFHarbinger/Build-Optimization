"""
GENIUS (GENI + US) configuration for Hydra.

Reference:
    Gendreau, M., Hertz, A., & Laporte, G. (1992).
    "New Insertion and Postoptimization Procedures for the Traveling Salesman Problem"
    Operations Research, 40(6), 1086-1094.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class GENIUSConfig:
    """Configuration for the GENIUS (GENI + US) policy."""

    neighborhood_size: int = 5
    unstring_type: int = 1
    string_type: int = 1
    n_iterations: int = 1
    random_us_sampling: bool = False
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    mandatory_selection: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
