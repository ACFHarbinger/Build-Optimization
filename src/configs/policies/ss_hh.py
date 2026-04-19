"""
SS-HH (Sequence-based Selection Hyper-Heuristic) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class SSHHConfig:
    """
    Configuration for the Sequence-based Selection Hyper-Heuristic policy.

    Reference: Kheiri (2014), Algorithm 1.

    Attributes:
        max_iterations: Total main-loop steps.
        n_removal: Nodes removed per LLH destroy step.
        n_llh: LLH pool size (fixed at 5).
        time_limit: Wall-clock time limit in seconds.
        threshold_infeasible: Acceptance threshold T when infeasible (Eq. 4).
        threshold_feasible_base: Base acceptance threshold T (Eq. 4).
        threshold_decay_rate: Time-decay coefficient for T (Eq. 4).
        vrpp: If True, solver operates in full VRPP mode.
        mandatory_selection: Mandatory selection strategy config list.
        route_improvement: Route improvement operation config list.
    """

    max_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
    threshold_infeasible: float = 0.001
    threshold_feasible_base: float = 0.0001
    threshold_decay_rate: float = 0.01
    seed: Optional[int] = None
    vrpp: bool = True
    mandatory_selection: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
