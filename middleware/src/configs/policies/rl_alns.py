"""
RL-ALNS (Reinforcement Learning-augmented Adaptive Large Neighborhood Search) configuration.
"""

from dataclasses import dataclass, field

from .alns import ALNSConfig
from .other import RLConfig


@dataclass
class RLALNSConfig(ALNSConfig):
    """Configuration for Reinforcement Learning-augmented ALNS (RL-ALNS) policy.

    Attributes:
        rl_config: Centralized Reinforcement Learning configuration.
        vrpp: Whether the problem is a VRP with Profits.
    """

    rl_config: RLConfig = field(default_factory=RLConfig)
    vrpp: bool = True
    profit_aware_operators: bool = False
