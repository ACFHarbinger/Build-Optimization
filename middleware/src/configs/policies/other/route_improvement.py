"""
Route improvement configuration module.

Defines structured configurations for route refinement and route improvement strategies,
mirroring the reinforcement learning configuration pattern.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .acceptance_criteria import AcceptanceConfig, BoltzmannAcceptanceConfig


@dataclass
class FastTSPPostConfig:
    """Configuration for Fast TSP optimization refinement.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        seed: Random seed for reproducibility.
    """

    time_limit: float = 2.0
    seed: int = 42


@dataclass
class LKHPostConfig:
    """Configuration for Lin-Kernighan-Helsgaun (LKH) refinement.

    Attributes:
        max_iterations: Maximum number of LKH iterations.
        time_limit: Maximum time in seconds for the solver.
        seed: Random seed for reproducibility.
    """

    max_iterations: int = 1000
    time_limit: float = 30.0
    seed: int = 42


@dataclass
class LocalSearchPostConfig:
    """Configuration for classical local search refinement.

    Attributes:
        ls_operator: Local search operator (e.g., '2opt', 'swap', 'relocate').
        iterations: Maximum number of local search iterations.
        time_limit: Maximum time in seconds for the operator.
        seed: Random seed for reproducibility.
    """

    ls_operator: str = "2opt"
    iterations: int = 1000
    time_limit: float = 30.0
    seed: int = 42


@dataclass
class PathPostConfig:
    """Configuration for path-based refinement (opportunistic pickups).

    Attributes:
        vehicle_capacity: Maximum vehicle capacity for picking up extra nodes.
    """

    vehicle_capacity: float = 100.0


@dataclass
class RandomLocalSearchPostConfig:
    """Configuration for stochastic (random) local search refinement.

    Attributes:
        iterations: Number of random local search iterations.
        params: Probabilities for selecting different local search operators.
        time_limit: Maximum time in seconds for the operator.
        seed: Random seed for reproducibility.
    """

    iterations: int = 1000
    params: Dict[str, float] = field(
        default_factory=lambda: {
            "two_opt": 0.2,
            "two_opt_star": 0.2,
            "swap": 0.15,
            "swap_star": 0.15,
            "relocate": 0.15,
            "three_opt": 0.1,
        }
    )
    time_limit: float = 30.0
    seed: int = 42


@dataclass
class OrOptPostConfig:
    """Configuration for Or-opt refinement."""

    chain_len: int = 2
    iterations: int = 500
    seed: int = 42


@dataclass
class CrossExchangePostConfig:
    """Configuration for Cross-exchange refinement."""

    cross_exchange_max_segment_len: int = 3
    iterations: int = 500
    seed: int = 42


@dataclass
class GuidedLocalSearchPostConfig:
    """Configuration for Guided Local Search."""

    gls_iterations: int = 20
    gls_inner_iterations: int = 50
    gls_lambda_factor: float = 0.1
    gls_base_operator: str = "or_opt"
    seed: int = 42


@dataclass
class SimulatedAnnealingPostConfig:
    """Configuration for Simulated Annealing refinement."""

    acceptance: AcceptanceConfig = field(
        default_factory=lambda: AcceptanceConfig(
            method="bmc", params=BoltzmannAcceptanceConfig(initial_temp=1.0, alpha=0.995)
        )
    )
    iterations: int = 2000
    params: Dict[str, float] = field(
        default_factory=lambda: {
            "two_opt": 0.2,
            "two_opt_star": 0.15,
            "swap": 0.15,
            "swap_star": 0.15,
            "relocate": 0.15,
            "three_opt": 0.1,
            "or_opt": 0.05,
            "cross_exchange": 0.05,
        }
    )
    seed: int = 42


@dataclass
class InsertionPostConfig:
    """Configuration for insertion-based augmentation strategies."""

    cost_per_km: float = 0.0
    revenue_kg: float = 0.0
    regret_k: int = 2
    detour_epsilon: float = 0.2
    n_bins: Optional[int] = None
    seed: int = 42


@dataclass
class RuinRecreatePostConfig:
    """Configuration for Ruin and Recreate (LNS)."""

    acceptance: AcceptanceConfig = field(default_factory=lambda: AcceptanceConfig(method="oi"))
    lns_iterations: int = 100
    ruin_fraction: float = 0.2
    repair_k: int = 2
    cost_per_km: float = 0.0
    revenue_kg: float = 0.0
    seed: int = 42


@dataclass
class AdaptiveLNSPostConfig:
    """Configuration for Adaptive LNS."""

    acceptance: AcceptanceConfig = field(default_factory=lambda: AcceptanceConfig(method="bmc"))
    alns_iterations: int = 200
    ruin_fraction: float = 0.2
    alns_bandit_warm_start_path: Optional[str] = None
    alns_ruin_ops: List[str] = field(default_factory=lambda: ["random", "worst", "shaw", "cluster"])
    alns_repair_ops: List[str] = field(default_factory=lambda: ["greedy", "regret"])
    repair_k: int = 2
    cost_per_km: float = 0.0
    revenue_kg: float = 0.0
    seed: int = 42


@dataclass
class FixAndOptimizePostConfig:
    """Configuration for Fix-and-Optimize refinement."""

    fo_n_free: Optional[int] = None
    fo_free_fraction: float = 0.30
    fo_time_limit: float = 30.0
    seed: int = 42


@dataclass
class SetPartitioningPostConfig:
    """Configuration for Set-partitioning (with pool construction)."""

    sp_n_perturbations: int = 20
    sp_include_dp: bool = True
    sp_time_limit: float = 60.0
    ruin_fraction: float = 0.2
    seed: int = 42


@dataclass
class SetPartitioningPolishPostConfig:
    """Configuration for Set-partitioning polish (bare wrapper)."""

    route_pool: Optional[List[List[int]]] = None
    sp_time_limit: float = 60.0
    seed: int = 42


@dataclass
class LearnedPostConfig:
    """Configuration for the learned route improver."""

    learned_weights_path: Optional[str] = None
    learned_max_iter: int = 100
    learned_min_improvement: float = 1e-4
    learned_neighborhood_size: int = 20
    seed: int = 42


@dataclass
class BranchAndPricePostConfig:
    """Configuration for Branch-and-price (Consolidated)."""

    bp_max_iterations: int = 100
    bp_max_routes_per_iteration: int = 10
    bp_optimality_gap: float = 1e-4
    bp_branching_strategy: str = "edge"
    bp_max_branch_nodes: int = 1000
    bp_use_exact_pricing: bool = True  # route improver default
    bp_use_ng_routes: bool = True
    bp_ng_neighborhood_size: int = 8
    bp_tree_search_strategy: str = "best_first"
    bp_vehicle_limit: Optional[int] = None
    bp_cleanup_frequency: int = 20
    bp_cleanup_threshold: float = -100.0
    bp_early_termination_gap: float = 1e-3
    bp_allow_heuristic_ryan_foster: bool = False
    bp_time_limit: float = 120.0
    bp_use_cspy: bool = True  # only used by vrpy fallback
    seed: int = 42


@dataclass
class MultiPhasePostConfig:
    """Configuration for Multi-phase composition."""

    phases: List[str] = field(default_factory=lambda: ["cheapest_insertion", "lkh"])
    seed: int = 42


@dataclass
class RouteImprovingConfig:
    """Unified configuration for route refinement and route improvement strategies.

    Composes algorithm-specific parameters and execution settings into a single object.

    Attributes:
        methods: List of route improvement methods to apply in sequence.
            Supported: 'fast_tsp', 'lkh', 'classical_local_search', 'random_local_search', 'path',
                       'or_opt', 'cross_exchange', 'guided_local_search', 'simulated_annealing',
                       'cheapest_insertion', 'regret_k_insertion', 'profitable_detour',
                       'ruin_recreate', 'adaptive_lns', 'multi_phase', 'steepest_two_opt',
                       'or_opt_steepest', 'node_exchange_steepest', 'dp_route_reopt',
                       'fix_and_optimize', 'set_partitioning_polish', 'set_partitioning',
                       'branch_and_price'.
        fast_tsp: Configuration for Fast TSP solver.
        lkh: Configuration for Lin-Kernighan-Helsgaun solver.
        local_search: Configuration for classical local search.
        random_local_search: Configuration for random local search.
        path: Configuration for path-based refinement.
        or_opt: Configuration for Or-opt strategy.
        cross_exchange: Configuration for Cross-exchange strategy.
        guided_local_search: Configuration for GLS strategy.
        simulated_annealing: Configuration for SA strategy.
        insertion: Shared configuration for insertion/augmentation strategies.
        ruin_recreate: Configuration for LNS.
        adaptive_lns: Configuration for ALNS.
        multi_phase: Configuration for Multi-phase composition.
        time_limit: Soft global time limit for route improvement operations.
        params: Additional strategy-specific parameters as a dictionary.
    """

    methods: List[str] = field(default_factory=lambda: ["fast_tsp"])

    # Algorithm-specific sub-configs
    fast_tsp: FastTSPPostConfig = field(default_factory=FastTSPPostConfig)
    lkh: LKHPostConfig = field(default_factory=LKHPostConfig)
    local_search: LocalSearchPostConfig = field(default_factory=LocalSearchPostConfig)
    random_local_search: RandomLocalSearchPostConfig = field(default_factory=RandomLocalSearchPostConfig)
    path: PathPostConfig = field(default_factory=PathPostConfig)

    # New sub-configs
    or_opt: OrOptPostConfig = field(default_factory=OrOptPostConfig)
    cross_exchange: CrossExchangePostConfig = field(default_factory=CrossExchangePostConfig)
    guided_local_search: GuidedLocalSearchPostConfig = field(default_factory=GuidedLocalSearchPostConfig)
    simulated_annealing: SimulatedAnnealingPostConfig = field(default_factory=SimulatedAnnealingPostConfig)
    insertion: InsertionPostConfig = field(default_factory=InsertionPostConfig)
    ruin_recreate: RuinRecreatePostConfig = field(default_factory=RuinRecreatePostConfig)
    adaptive_lns: AdaptiveLNSPostConfig = field(default_factory=AdaptiveLNSPostConfig)
    multi_phase: MultiPhasePostConfig = field(default_factory=MultiPhasePostConfig)

    # Added strategies
    fix_and_optimize: FixAndOptimizePostConfig = field(default_factory=FixAndOptimizePostConfig)
    set_partitioning: SetPartitioningPostConfig = field(default_factory=SetPartitioningPostConfig)
    set_partitioning_polish: SetPartitioningPolishPostConfig = field(default_factory=SetPartitioningPolishPostConfig)
    branch_and_price: BranchAndPricePostConfig = field(default_factory=BranchAndPricePostConfig)
    learned: LearnedPostConfig = field(default_factory=LearnedPostConfig)

    # Generic operator parameters
    max_iter: int = 500
    dp_max_nodes: int = 20
    chain_lengths: List[int] = field(default_factory=lambda: [1, 2, 3])

    # Additional parameters
    params: Dict[str, Any] = field(default_factory=dict)
