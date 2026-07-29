"""
RL-HVPL (Reinforcement Learning Hybrid Volleyball Premier League) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .aco_ks import KSparseACOConfig
from .alns import ALNSConfig
from .other import RLConfig


@dataclass
class RLHVPLConfig:
    """
    Configuration for the Reinforcement Learning Hybrid Volleyball Premier League policy.

    Combines VPL population dynamics, ACO initialization with Q-Learning,
    and ALNS local search with SARSA for adaptive operator selection.
    """

    # General parameters
    n_teams: int = 10
    time_limit: float = 60.0
    max_iterations: int = 100
    elite_coaching_iterations: int = 300
    regular_coaching_iterations: int = 100
    elite_size: int = 3
    sub_rate: float = 0.2
    pheromone_update_strategy: str = "profit"
    profit_weight: float = 1.0

    # Nested component configs
    aco: KSparseACOConfig = field(default_factory=KSparseACOConfig)
    alns: ALNSConfig = field(default_factory=ALNSConfig)

    # RL Configuration (Centralized)
    rl_config: RLConfig = field(default_factory=RLConfig)

    # Common policy fields
    vrpp: bool = True
    seed: Optional[int] = None
    profit_aware_operators: bool = False
    mandatory_selection: List[str] = field(default_factory=list)
    route_improvement: List[Any] = field(default_factory=list)
