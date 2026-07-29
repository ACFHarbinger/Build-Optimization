"""
GLS (Guided Local Search) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class GLSConfig:
    """Configuration for the Guided Local Search policy."""

    lambda_param: float = 1.0
    alpha_param: float = 0.3
    penalty_cycles: int = 1000
    n_removal: int = 2
    n_llh: int = 6
    inner_iterations: int = 100
    fls_coupling_prob: float = 0.8
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False
    mandatory_selection: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
