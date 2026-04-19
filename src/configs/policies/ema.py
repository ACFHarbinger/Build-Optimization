"""
EMA (Ensemble Move Acceptance) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EMAConfig:
    """Configuration for the Ensemble Move Acceptance policy."""

    max_iterations: int = 1000
    rule: str = "G-VOT"
    criteria: List[str] = field(default_factory=lambda: ["sa", "gd", "ie"])
    sub_params: Dict[str, Any] = field(default_factory=dict)
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
    seed: Optional[int] = None
    profit_aware_operators: bool = True
    vrpp: bool = True
    mandatory_selection: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
