"""
Configuration parameters for the Adaptive Large Neighborhood Search (ALNS).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion


@dataclass
class ALNSParams:
    """
    Configuration parameters for the ALNS solver.

    Attributes:
        time_limit: Maximum runtime in seconds.
        max_iterations: Maximum number of ALNS iterations.
        start_temp: Initial temperature for simulated annealing (if > 0, dynamic calculation is disabled).
        cooling_rate: Temperature decay factor per iteration.
        reaction_factor: Learning rate for operator weight updates (r in Ropke & Pisinger 2005).
        min_removal: Minimum number of nodes to remove.
        max_removal_pct: Maximum percentage of nodes to remove.
        segment_size: Number of iterations before updating operator weights.
        noise_factor: Noise level for repair operators (eta in Ropke & Pisinger 2005).
        worst_removal_randomness: Randomness parameter p >= 1 for worst removal (p=1 is deterministic).
        sigma_1: Score for new global best solution.
        sigma_2: Score for better solution (not global best, not visited before).
        sigma_3: Score for accepted worse solution (not visited before).
        vrpp: If True, allow expanding insertion pool beyond removed nodes.
        profit_aware_operators: If True, use profit-aware insertion/removal operators.
            This flag enables ablation studies:
            - False: Standard ALNS operators (worst_removal, greedy_insertion, regret_k_insertion)
            - True: Profit-aware variants (worst_profit_removal, greedy_profit_insertion, regret_k_profit_insertion)
            For scientific publication, run comparative experiments with this flag toggled
            to statistically validate the contribution of profit-aware heuristics.
        extended_operators: If True, add string, cluster, and neighbor removal operators
            to the destroy pool in addition to random/worst/shaw (3 → 6 operators).
            Increases operator diversity at the cost of slightly higher per-iteration overhead.
        seed: Random seed for reproducibility.
        engine: Engine to use for the solver.
    """

    time_limit: float = 60.0
    max_iterations: int = 5000
    start_temp: float = 0.0  # 0 = dynamic calculation; > 0 = fixed start temperature
    cooling_rate: float = 0.995
    reaction_factor: float = 0.1  # r in w_{i,j+1} = w_{i,j}(1-r) + r * (π_i / θ_i)
    min_removal: int = 4  # Paper: 4 ≤ q ≤ min(100, ξn), Section 4.3.1
    max_removal_pct: float = 0.3
    segment_size: int = 100  # Iterations per segment (Ropke & Pisinger 2005)
    noise_factor: float = 0.025  # eta: noise level relative to max distance
    worst_removal_randomness: float = 3.0  # p >= 1: randomness in worst removal
    shaw_randomization: float = 6.0  # p_shaw in Ropke & Pisinger (2005)
    max_removal_cap: int = 100  # Hard upper cap on q: q <= min(100, xi * n)
    regret_pool: str = "regret234"  # "regret2", "regret234", or "regretAll"
    sigma_1: float = 33.0  # Score for new global best
    sigma_2: float = 9.0  # Score for improving solution (not visited)
    sigma_3: float = 13.0  # Score for accepted worse solution (not visited)
    vrpp: bool = True
    profit_aware_operators: bool = False
    extended_operators: bool = False
    seed: Optional[int] = None
    engine: str = "custom"
    acceptance_criterion: Optional[IAcceptanceCriterion] = None

    @classmethod
    def from_config(cls, config: Any) -> ALNSParams:
        """Create ALNSParams from an ALNSConfig dataclass or dict.

        Args:
            config: ALNSConfig dataclass or dict with solver parameters.

        Returns:
            ALNSParams instance with values from config.
        """
        if isinstance(config, dict):
            params = cls(**{k: v for k, v in config.items() if k in {f.name for f in dataclasses.fields(cls)}})
        else:
            params = cls(
                time_limit=config.time_limit,
                max_iterations=config.max_iterations,
                start_temp=config.start_temp,
                cooling_rate=config.cooling_rate,
                reaction_factor=config.reaction_factor,
                min_removal=getattr(config, "min_removal", 4),
                max_removal_pct=getattr(config, "max_removal_pct", 0.3),
                segment_size=getattr(config, "segment_size", 100),
                noise_factor=getattr(config, "noise_factor", 0.025),
                worst_removal_randomness=getattr(config, "worst_removal_randomness", 3.0),
                shaw_randomization=getattr(config, "shaw_randomization", 6.0),
                max_removal_cap=getattr(config, "max_removal_cap", 100),
                regret_pool=getattr(config, "regret_pool", "regret234"),
                sigma_1=getattr(config, "sigma_1", 33.0),
                sigma_2=getattr(config, "sigma_2", 9.0),
                sigma_3=getattr(config, "sigma_3", 13.0),
                vrpp=getattr(config, "vrpp", True),
                profit_aware_operators=getattr(config, "profit_aware_operators", False),
                extended_operators=getattr(config, "extended_operators", False),
                seed=getattr(config, "seed", None),
                engine=getattr(config, "engine", "custom"),
            )

        # Handle Acceptance Criterion Injection
        from logic.src.policies.route_construction.acceptance_criteria.base.factory import AcceptanceCriterionFactory

        acceptance_cfg = getattr(config, "acceptance", None)
        if acceptance_cfg:
            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name=acceptance_cfg.method,
                config=acceptance_cfg.params,
            )
        else:
            # Automatic mapping for legacy compatibility
            # Use Boltzmann with original start_temp and cooling_rate
            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name="bmc",
                initial_temp=params.start_temp,
                alpha=params.cooling_rate,
                seed=params.seed,
            )

        return params
