r"""
Configuration parameters for the Integer L-Shaped (Benders Decomposition) solver.

Based on:
    Laporte, G., & Louveaux, F. V. (1993). "The integer L-shaped method for
    stochastic integer programs with complete recourse". Operations Research
    Letters, 13(3), 133-142.

    Kleywegt, A. J., Shapiro, A., & Homem-de-Mello, T. (2002). "The sample
    average approximation method for stochastic discrete optimization". SIAM
    Journal on Optimization, 12(2), 479-502.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class ILSBDParams:
    r"""Configuration parameters for the Integer L-Shaped (Benders Decomposition) solver.

    The solver formulates the Stochastic IRP as a Two-Stage Stochastic Integer
    Program (2-SIP) and iteratively solves it via Benders decomposition:

        Stage 1 (Master Problem):
            Routing decisions (x_ij ∈ {0,1}) and node-visit decisions (y_i ∈ {0,1})
            plus a surrogate variable θ representing the expected recourse cost.

        Stage 2 (Recourse Subproblem):
            Expected penalty Q̄(ŷ) evaluated analytically across S discrete SAA
            scenarios of end-of-day bin fill levels.  No LP solve is required
            because the subproblem is separable per bin:
                Q(ŷ, ω) = Σᵢ [p⁺ᵢ(ω)·(1−ŷᵢ) + p⁻ᵢ(ω)·ŷᵢ]

    Convergence is declared when Q̄(ŷ̂) ≤ θ̂ + benders_gap, i.e., the surrogate
    faithfully represents the expected recourse cost.

    Attributes:
        time_limit: Maximum wall-clock seconds for the overall Benders solve.
        master_time_limit: Time limit for each master problem solve (seconds).
        n_scenarios: Number of SAA discrete scenarios for recourse approximation.
        seed: Random seed for reproducible scenario generation.
        vrpp: Whether to formulate as VRPP (some customer nodes may be skipped).
        profit_aware_operators: Whether to use profit-aware warm-start heuristics.
        max_benders_iterations: Maximum outer Benders (L-shaped) iterations.
        benders_gap: Convergence tolerance: |Q̄(ŷ) − θ| < benders_gap → optimal.
        overflow_penalty: Penalty per %-fill unit that overflows an unvisited bin.
            Represents the marginal cost of not emptying a bin whose fill exceeds
            the collection threshold τ (€ per %-fill above τ).
        undervisit_penalty: Penalty per %-fill unit below threshold at visited bins.
            Represents the wasted-trip marginal cost of visiting a nearly-empty bin
            (€ per %-fill below τ when bin is visited).
        collection_threshold: Fill level percentage τ at which a bin is considered
            "full". Overflow penalties are triggered for unvisited bins with
            fill > τ; undervisit penalties are triggered for visited bins with
            fill < τ.
        horizon: Planning horizon T (number of days).
        stockout_penalty: Penalty per unit of bin overflow per day.
        big_m: Big-M constant for decoupling collection quantities and visit variables.
        mean_scenario_only: If True, use mean demand for balance constraints in the master problem.
        initial_inventory: Initial fill level (%) for all bins (default 0).
        fill_rate_cv: Coefficient of variation for the Gamma-distribution used in
            scenario generation.  CV = σ/μ; higher values produce more spread.
        mip_gap: Relative MIP gap tolerance for each Gurobi master problem solve.
        theta_lower_bound: Initial lower bound imposed on the surrogate variable θ.
            Defaults to 0.0 (non-negative recourse); set negative if the problem
            formulation admits negative recourse contributions.
        verbose: Enable Gurobi solver output and Benders iteration logging.
        max_cuts_per_round: Maximum SECs / capacity cuts added per Gurobi callback.
        enable_heuristic_rcc_separation: Enable heuristic Rounded Capacity Cut
            (RCC) separation at fractional B&B nodes.
        enable_comb_cuts: Enable heuristic comb inequality separation.
    """

    time_limit: float = 300.0
    master_time_limit: float = 60.0
    n_scenarios: int = 20
    seed: Optional[int] = 42
    vrpp: bool = True
    profit_aware_operators: bool = False
    max_benders_iterations: int = 50
    benders_gap: float = 1e-4
    overflow_penalty: float = 100.0
    undervisit_penalty: float = 10.0
    collection_threshold: float = 70.0
    horizon: int = 7
    stockout_penalty: float = 500.0
    big_m: float = 1e4
    mean_scenario_only: bool = True
    initial_inventory: float = 0.0
    fill_rate_cv: float = 0.3
    mip_gap: float = 0.01
    theta_lower_bound: float = 0.0
    verbose: bool = False
    max_cuts_per_round: int = 50
    enable_heuristic_rcc_separation: bool = True
    enable_comb_cuts: bool = False
    engine: str = "gurobi"
    framework: str = "ortools"

    @classmethod
    def from_config(cls, config: Any) -> ILSBDParams:
        """Create ILSBDParams from a configuration object or dictionary.

        Args:
            config: Dict or object with named attributes corresponding to fields.

        Returns:
            Typed ILSBDParams instance with defaults for missing fields.
        """
        if isinstance(config, dict):
            valid_keys = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in config.items() if k in valid_keys})

        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            if f.default is not MISSING:
                kwargs[f.name] = getattr(config, f.name, f.default)
            elif f.default_factory is not MISSING:  # type: ignore[misc]
                kwargs[f.name] = getattr(config, f.name, f.default_factory())  # type: ignore[misc]
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ILSBDParams to a plain dictionary.

        Returns:
            Dict mapping field name to field value.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}
