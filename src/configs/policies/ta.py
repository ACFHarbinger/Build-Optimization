"""
TA (Threshold Accepting) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class TAConfig:
    """Configuration for the Threshold Accepting policy."""

    max_iterations: int = 1000
    initial_threshold: float = 100.0
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
    seed: Optional[int] = None
    profit_aware_operators: bool = True
    vrpp: bool = True
    mandatory_selection: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
