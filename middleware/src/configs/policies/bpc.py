"""
BPC (Branch-and-Price-and-Cut) configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class BPCConfig:
    """Configuration for Branch-and-Price-and-Cut (BPC) policy.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        profit_aware_operators: Whether to use profit-aware operators.
        vrpp: Whether to use VRPP.
        seed: Random seed for reproducibility.
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
        search_strategy: B&B node selection strategy ('best_first' or 'depth_first').
        cutting_planes: Cutting plane family ('rcc', 'sri', 'lci', or 'all').
        branching_strategy: Branching rule ('ryan_foster', 'edge', or 'divergence').
        max_cg_iterations: Maximum iterations for column generation loop.
        max_cut_iterations: Maximum iterations for cutting plane loop per CG iteration.
        max_cuts_per_iteration: Maximum cuts to add per iteration.
        max_routes_per_pricing: Maximum routes to add per pricing call.
        max_bb_nodes: Maximum nodes to explore in the B&B tree.
        optimality_gap: Relative gap for proven optimality.
        early_termination_gap: Gap at which to stop search early (heuristic shortcut).
        use_ng_routes: Whether to use ng-route relaxation in pricing.
        ng_neighborhood_size: Size of ng-neighborhoods for relaxation.
        enable_heuristic_rcc_separation: Whether to enable (heuristic) fractional RCC separation.
        enable_comb_cuts: Whether to enable heuristic comb inequalities.
        cut_orthogonality_threshold: Cosine similarity ceiling for cut filtering.
        use_spatial_partitioning: Whether to use spatial partitioning for branching.
        enable_strong_branching_heuristic: Whether to enable heuristic strong branching.
        enable_column_pool_deduplication: Whether to enable column pool deduplication.
        enable_hybrid_search: Whether to enable hybrid DFS/BFS search.
        rc_tolerance: Minimum reduced cost to accept a new column.
        exact_mode: Whether to enable strict exact management.
        strong_branching_size: Number of candidates for strong branching.
        cg_at_root_only: Run column generation only at the root B&B node; at all
            descendant nodes, no new columns are generated (use the pool built at
            root).  Replicates the "CG at root only" experiment in Table 2 of
            Barnhart, Hane, and Vance (2000).  Trades solution quality for speed.
        use_swc_tcf_initialization: Whether to use SWC-TCF for RMP initialization.
        use_swc_tcf_heuristic_pricing: Whether to use SWC-TCF as a fast heuristic pricer.
        use_swc_tcf_primal_heuristic: Whether to use SWC-TCF as a primal heuristic at nodes.
    """

    time_limit: float = 60.0
    profit_aware_operators: bool = False
    vrpp: bool = True
    seed: Optional[int] = None
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
    search_strategy: str = "depth_first"
    cutting_planes: str = "rcc"
    branching_strategy: str = "divergence"
    max_cg_iterations: int = 50
    max_cut_iterations: int = 5
    max_cuts_per_iteration: int = 5
    max_routes_per_pricing: int = 5
    max_bb_nodes: int = 1000
    optimality_gap: float = 1e-4
    early_termination_gap: float = 1e-3
    use_ng_routes: bool = True
    ng_neighborhood_size: int = 8
    enable_heuristic_rcc_separation: bool = True
    enable_comb_cuts: bool = False
    cut_orthogonality_threshold: float = 0.8
    use_spatial_partitioning: bool = False
    enable_strong_branching_heuristic: bool = True
    enable_column_pool_deduplication: bool = True
    enable_hybrid_search: bool = False
    rc_tolerance: float = 1e-5
    exact_mode: bool = False
    strong_branching_size: int = 5
    cg_at_root_only: bool = False
    use_swc_tcf_initialization: bool = False
    use_swc_tcf_heuristic_pricing: bool = False
    use_swc_tcf_primal_heuristic: bool = False
    lr_pre_pruning: bool = False
    lr_lambda_init: float = 0.0
    lr_max_subgradient_iters: int = 30
    lr_subgradient_theta: float = 1.0
    lr_op_time_limit: float = 3.0
    lr_pre_pruning_depth_limit: int = -1
    lr_warm_start_cg: bool = False
    multi_day_mode: bool = False
    adp_model_path: str = ""
    adp_model_type: str = "sklearn"
