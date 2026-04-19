"""
TS (Tabu Search) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from logic.src.configs.policies.other.acceptance_criteria import AcceptanceConfig


@dataclass
class TSConfig:
    """Configuration for the Tabu Search policy."""

    # Short-term memory (Recency-based)
    tabu_tenure: int = 7
    dynamic_tenure: bool = True
    min_tenure: int = 5
    max_tenure: int = 15

    # Aspiration criteria
    aspiration_enabled: bool = True

    # Long-term memory (Frequency-based)
    intensification_enabled: bool = True
    diversification_enabled: bool = True
    intensification_interval: int = 100
    diversification_interval: int = 200
    elite_size: int = 5
    frequency_penalty_weight: float = 0.1

    # Candidate list strategies
    candidate_list_enabled: bool = True
    candidate_list_size: int = 20

    # Strategic oscillation
    oscillation_enabled: bool = False
    feasibility_tolerance: float = 0.1

    # General search parameters
    max_iterations: int = 5000
    max_iterations_no_improve: int = 500
    n_removal: int = 3
    n_llh: int = 5
    time_limit: float = 60.0

    # Neighborhood structure
    use_swap: bool = True
    use_relocate: bool = True
    use_2opt: bool = True
    use_insertion: bool = True

    # Standard fields
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False
    mandatory_selection: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)

    # Injected Acceptance Criterion
    acceptance: AcceptanceConfig = field(default_factory=lambda: AcceptanceConfig(method="tabu_aspiration"))
