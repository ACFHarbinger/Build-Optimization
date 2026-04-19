"""
AHVPL (Augmented Hybrid Volleyball Premier League) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .aco_ks import KSparseACOConfig
from .alns import ALNSConfig
from .hgs import HGSConfig


@dataclass
class AHVPLConfig:
    """
    Configuration for the Augmented Hybrid Volleyball Premier League policy.

    Extends HVPL with HGS integration parameters for diversity-driven
    crossover and bi-criteria fitness evaluation.
    """

    # VPL League Parameters
    n_teams: int = 10
    max_iterations: int = 50
    sub_rate: float = 0.2
    time_limit: float = 60.0
    alns_elite_iterations: int = 500
    alns_not_coached_iterations: int = 100
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    # Nested component configs
    hgs: HGSConfig = field(default_factory=HGSConfig)
    aco: KSparseACOConfig = field(default_factory=KSparseACOConfig)
    alns: ALNSConfig = field(default_factory=ALNSConfig)

    # Common policy fields
    mandatory_selection: List[str] = field(default_factory=list)
    route_improvement: List[Any] = field(default_factory=list)
