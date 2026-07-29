"""
HMM-GD (Hidden Markov Model + Great Deluge) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class HMMGDHHConfig:
    """
    Configuration for the Hidden Markov Model + Great Deluge Hyper-Heuristic policy.

    Attributes:
        max_iterations: Total LLH applications.
        flood_margin: Initial water level offset as fraction of initial profit.
        rain_speed: Rate of water level decrease per iteration.
        learning_rate: Online HMM transition probability update step.
        n_removal: Nodes removed per destroy step.
        n_llh: LLH pool size (fixed at 5).
        local_search_iterations: Number of local search iterations.
        time_limit: Wall-clock time limit in seconds.
        vrpp: If True, solver operates in full VRPP mode.
        mandatory_selection: Mandatory selection strategy config list.
        route_improvement: Route improvement operation config list.
    """

    max_iterations: int = 500
    flood_margin: float = 0.05
    rain_speed: float = 0.001
    learning_rate: float = 0.1
    n_removal: int = 2
    n_llh: int = 5
    local_search_iterations: int = 500
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False
    mandatory_selection: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
