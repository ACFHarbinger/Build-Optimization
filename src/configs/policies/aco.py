"""
ACO (Ant Colony Optimization) configuration.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ACOConfig:
    engine: str = "aco"
    n_ants: int = 20
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    tau_0: float = 1.0
    tau_min: float = 0.01
    tau_max: float = 10.0
    max_iterations: int = 50
    time_limit: float = 30.0
    q0: float = 0.9
    k_sparse: int = 15
    sequence_length: int = 5
    local_search: bool = True
    elitist_weight: float = 1.0
    operators: List[str] = field(default_factory=lambda: ["swap", "2opt_intra", "relocate", "swap_star", "perturb"])
