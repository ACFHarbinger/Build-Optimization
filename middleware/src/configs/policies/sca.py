"""
SCA (Sine Cosine Algorithm) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class SCAConfig:
    """
    Configuration for the Sine Cosine Algorithm policy.

    Attributes:
        pop_size: Population size.
        a_max: Initial control parameter value (linearly decays to 0).
        max_iterations: Maximum SCA iterations.
        time_limit: Wall-clock time limit in seconds.
        vrpp: If True, solver operates in full VRPP mode.
        mandatory_selection: Mandatory selection strategy config list.
        route_improvement: Route improvement operation config list.
    """

    pop_size: int = 20
    a_max: float = 2.0
    max_iterations: int = 200
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    mandatory_selection: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
