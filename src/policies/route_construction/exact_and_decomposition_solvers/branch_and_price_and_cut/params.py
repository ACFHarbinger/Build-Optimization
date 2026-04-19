"""
Configuration parameters for the Branch-and-Price-and-Cut (BPC) solver.

Based on Barnhart et al. (1998, 2000) and standard exact VRPP protocols.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class BPCParams:
    """
    Configuration parameters for the Branch-and-Price-and-Cut solver.

    Attributes:
        optimality_gap: Relative gap for proven optimality.
        early_termination_gap: Gap at which we stop the search (heuristic shortcut).
        use_ng_routes: Whether to use ng-route relaxation in pricing.
        ng_neighborhood_size: Size of ng-neighborhoods for relaxation.
        enable_heuristic_rcc_separation: Whether to enable (heuristic) fractional RCC separation.
        enable_comb_cuts: Whether to enable heuristic comb inequalities.
        cut_orthogonality_threshold: cosine similarity ceiling for cut filtering.
        enable_column_pool_deduplication: Whether to enable column pool deduplication.
        rc_tolerance: Minimum reduced cost to accept a new column.
        exact_mode: Whether to enable strict exact management (no dual smoothing).
        strong_branching_size: Number of candidates to evaluate in strong branching.
        cg_at_root_only: Run column generation only at the root B&B node; at all
            descendant nodes, no new columns are generated (use the pool built at
            root).  Replicates the "CG at root only" experiment in Table 2 of
            Barnhart, Hane, and Vance (2000).  Trades solution quality for speed.


    Note:
        enable_heuristic_rcc_separation replaces the legacy enable_fractional_capacity_cuts
        to clarify that the connected-component-based separation is a heuristic,
        not an exact max-flow separation on the full support graph.
    """

    time_limit: float = 60.0
    profit_aware_operators: bool = False
    vrpp: bool = True
    seed: Optional[int] = None
    search_strategy: str = "depth_first"
    cutting_planes: str = "rcc"
    branching_strategy: str = "divergence"
    max_cg_iterations: int = 50
    max_cut_iterations: int = 5
    max_cuts_per_iteration: int = 5
    max_routes_per_pricing: int = 5
    max_bb_nodes: int = 1000
    optimality_gap: float = 1e-4
    early_termination_gap: float = 1e-3  # HEURISTIC SHORTCUT: See docstring for caveats.
    use_ng_routes: bool = True
    ng_neighborhood_size: int = 8
    enable_heuristic_rcc_separation: bool = True
    enable_comb_cuts: bool = False
    cut_orthogonality_threshold: float = 0.8
    use_spatial_partitioning: bool = False
    # Disabled by default. The current implementation is a heuristic proxy
    # that returns the top pre-sorted divergence candidate without solving
    # any child LP relaxations. When True, it bypasses fleet-size and
    # Ryan-Foster branching levels at every fractional node for no benefit.
    # Enable only after implementing full LP-based strong branching
    # (2×N child LP solves per candidate).
    enable_strong_branching_heuristic: bool = False
    enable_column_pool_deduplication: bool = True
    rc_tolerance: float = 1e-8  # Fix 8: Minimum reduced cost to accept a new column
    exact_mode: bool = False  # Task 3: Enable strict exact management
    strong_branching_size: int = 5  # Task 1: Number of candidates for strong branching
    cg_at_root_only: bool = False  # Paper Table 2: Run column generation only at root node

    # Matheuristic Integration Flags (SWC-TCF)
    use_swc_tcf_initialization: bool = False
    use_swc_tcf_heuristic_pricing: bool = False
    use_swc_tcf_primal_heuristic: bool = False

    # Multi-period adaptation and Adaptive Dynamic Programming Machine Learning model
    multi_day_mode: bool = False
    adp_model_path: str = ""
    adp_model_type: str = "sklearn"

    # ---------------------------------------------------------------------------
    # Lagrangian Relaxation Pre-Pruning
    # ---------------------------------------------------------------------------
    # When enabled, a fast subgradient pass is run at each B&B node before column
    # generation. If the Lagrangian bound is dominated by the incumbent, the node
    # is pruned without touching the master LP.
    #
    # Theoretical basis: L(λ*) ≥ z* (VRPP optimum) for all λ ≥ 0. If the CG LP
    # bound is the true Lagrangian dual (as proven by Dantzig-Wolfe equivalence),
    # then for tightly converged nodes the LR bound adds nothing. However at early
    # B&B nodes — especially in deep subtrees with heavy branching fixings — a
    # fast LR pass over the reduced customer set can prune before CG starts.
    # ---------------------------------------------------------------------------
    lr_pre_pruning: bool = False
    """Enable Lagrangian pre-pruning at B&B nodes before column generation."""

    lr_lambda_init: float = 0.0
    """Initial Lagrange multiplier λ₀ ≥ 0 for the capacity constraint."""

    lr_max_subgradient_iters: int = 30
    """Max Polyak iterations per node. Keep low (20–40) since this runs at every node."""

    lr_subgradient_theta: float = 1.0
    """Step-size multiplier θ ∈ (0, 2] for the Polyak rule."""

    lr_op_time_limit: float = 3.0
    """Per-solve wall-clock budget (seconds) for the uncapacitated OP inner solver."""

    lr_pre_pruning_depth_limit: int = -1
    """Maximum B&B tree depth at which to apply LR pruning.
    -1 means apply at every depth. Set to 0 to restrict to root only.
    Useful because LR bounds weaken at deep nodes due to many forced-in/out fixings."""

    lr_warm_start_cg: bool = False
    """If True, use λ* from subgradient to seed initial columns before CG starts.
    Calls solve_uncapacitated_op at λ* and adds the resulting route to the column
    pool. Provides a stronger starting point for CG at the cost of one extra OP solve.
    Only meaningful when lr_pre_pruning is also True (shares the λ* already computed)."""

    @classmethod
    def from_config(cls, config: Any) -> BPCParams:
        """Create BPCParams from a configuration object or dictionary."""
        if isinstance(config, dict):
            valid_keys = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in config.items() if k in valid_keys})  # type: ignore[arg-type]

        kwargs: Dict[str, Any] = {
            f.name: getattr(config, f.name, f.default) for f in fields(cls) if f.default is not MISSING
        }
        # Fields with default_factory
        for f in fields(cls):
            if f.name not in kwargs:
                kwargs[f.name] = f.default_factory()  # type: ignore[misc]
        return cls(**kwargs)  # type: ignore[arg-type]

    def to_dict(self) -> Dict[str, Any]:
        """Convert BPCParams to a dictionary for backend compatibility."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
