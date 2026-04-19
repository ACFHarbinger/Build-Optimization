"""
VNS (Variable Neighborhood Search) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from logic.src.configs.policies.other.acceptance_criteria import AcceptanceConfig


@dataclass
class VNSConfig:
    """Configuration for the Variable Neighborhood Search policy."""

    k_max: int = 5
    max_iterations: int = 200
    local_search_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False
    mandatory_selection: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)

    # Injected Acceptance Criterion
    acceptance: AcceptanceConfig = field(default_factory=lambda: AcceptanceConfig(method="only_improving"))
