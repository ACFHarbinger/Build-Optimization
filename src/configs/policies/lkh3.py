"""
LKH-3 (Lin-Kernighan-Helsgaun 3) configuration for Hydra.

Reference:
    Helsgaun, K. (2017). An extension of the LKH-TSP solver for constrained
    traveling salesman and vehicle routing problems.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class LKH3Config:
    """Configuration for the LKH-3 policy.

    Attributes:
        max_trials: Maximum number of local-search iterations per run.
        runs: Number of independent LKH runs (multi-start).
        popmusic_subpath_size: Sub-path size for POPMUSIC candidate generation.
        popmusic_trials: Number of POPMUSIC decomposition runs.
        max_k_opt: Maximum k for k-opt moves (2–5).
        use_ip_merging: If True, use IP-based tour merging; else greedy.
        time_limit: Wall-clock time limit in seconds.
        seed: Random seed for reproducibility.
        vrpp: If True, solver operates in full VRPP mode.
        dynamic_topology_discovery: If True, solver performs dynamic topology discovery.
        native_prize_collecting: If True, solver uses native prize collecting.
        mandatory_selection: Mandatory selection strategy config list.
        route_improvement: Route improvement operation config list.
    """

    runs: int = 10
    max_trials: int = 1000
    popmusic_subpath_size: int = 50
    popmusic_trials: int = 50
    popmusic_max_candidates: int = 5
    max_k_opt: int = 5
    use_ip_merging: bool = True
    max_pool_size: int = 5
    subgradient_iterations: int = 50  # Held-Karp ascent iterations for alpha-measure
    profit_aware_operators: bool = False
    alns_iterations: int = 100
    plateau_limit: int = 10
    deep_plateau_limit: int = 30
    perturb_operator_weights: List[float] = field(default_factory=lambda: [0.6, 0.4])
    time_limit: float = 60.0
    vrpp: bool = True
    dynamic_topology_discovery: bool = False
    native_prize_collecting: bool = False
    seed: Optional[int] = None

    mandatory_selection: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
