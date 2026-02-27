"""
AHVPL configuration.
"""

from dataclasses import dataclass, field

from .aco import ACOConfig
from .alns import ALNSConfig
from .hgs import HGSConfig


@dataclass
class AHVPLConfig:
    engine: str = "ahvpl"
    n_teams: int = 10
    max_iterations: int = 50
    sub_rate: float = 0.2
    time_limit: float = 60.0
    hgs: HGSConfig = field(default_factory=HGSConfig)
    aco: ACOConfig = field(default_factory=ACOConfig)
    alns: ALNSConfig = field(default_factory=ALNSConfig)
