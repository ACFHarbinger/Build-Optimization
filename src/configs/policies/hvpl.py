"""
HVPL (Hybrid Volleyball Premier League) configuration.
"""

from dataclasses import dataclass, field

from .aco import ACOConfig
from .alns import ALNSConfig


@dataclass
class HVPLConfig:
    engine: str = "hvpl"
    n_teams: int = 10
    max_iterations: int = 50
    sub_rate: float = 0.2
    time_limit: float = 60.0
    aco: ACOConfig = field(default_factory=ACOConfig)
    alns: ALNSConfig = field(default_factory=ALNSConfig)
