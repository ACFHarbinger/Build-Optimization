"""
RTS (Reactive Tabu Search) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .other.acceptance_criteria import AcceptanceConfig, AspirationConfig


@dataclass
class RTSConfig:
    """Configuration for the Reactive Tabu Search policy."""

    initial_tenure: int = 7
    min_tenure: int = 3
    max_tenure: int = 20
    tenure_increase: float = 1.5
    tenure_decrease: float = 0.9
    max_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False
    mandatory_selection: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
    acceptance: AcceptanceConfig = AcceptanceConfig(method="ac", params=AspirationConfig())
