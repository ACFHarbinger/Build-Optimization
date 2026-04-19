"""
Configuration parameters for the RL-GD-HH solver.

This module defines the hyper-parameters for the Reinforcement Learning –
Great-Deluge Hyper-heuristic, mapping them directly to the terminology and
experimental settings found in Ozcan et al. (2010).

Paper mappings (Section 3 and Table 1):
    - maxUtilityValue = utility_upper_bound (default 40, best of {20,40,60,80})
    - Initial utility = 0.75 × maxUtilityValue = 30
    - qualityLB = quality_lb (= 0 for all benchmark instances in the paper)
    - totalTime = time_limit (600 s in the paper; lower for VRPP instances)
    - Reward magnitude = +1 (RL1 additive)
    - Penalty magnitude = -1 (RL1 subtractive), ÷2 (RL2), √ (RL3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion
from logic.src.policies.route_construction.acceptance_criteria.base.factory import AcceptanceCriterionFactory


@dataclass
class RLGDHHParams:
    """
    Hyper-parameters for the RL-GD-HH solver.

    All defaults reproduce the best-performing configuration reported in
    Ozcan et al. (2010) unless otherwise noted.

    Attributes:
        max_iterations: Hard cap on hyper-heuristic search steps.
            Used as the iteration proxy for the time-based GD formula.
        time_limit: Optional wall-clock guard (seconds). Set ≤ 0 to disable.
        seed: RNG seed for reproducibility.

        reward_improvement: Additive reward (+1 per paper, Step 14) applied
            when a LLH strictly improves the current solution.
        penalty_worsening: Magnitude of the penalty (−1 per paper, Step 17)
            applied when a LLH fails to strictly improve (including neutral moves).
        punishment_type: Selects the punishment variant from the paper
            (Section 3.2):
            - "RL1": subtractive,  u ← max(0, u − penalty_worsening)
            - "RL2": divisional,   u ← floor(u / 2)
            - "RL3": root,         u ← floor(sqrt(u))
        utility_upper_bound: Maximum allowed utility (maxUtilityValue = 40).
        min_utility: Minimum allowed utility (= 0 per paper; prevents a
            heuristic from being permanently discarded).
        initial_utility: Starting utility for every LLH.
            Paper specifies 0.75 × maxUtilityValue = 0.75 × 40 = 30.
        quality_lb: The floor of the Great Deluge level (qualityLB = 0 for
            all Toronto/Yeditepe instances in the paper). The water level
            declines from f₀ to quality_lb over the full search budget.

        vrpp: Whether the problem is a VRPP (affects operator pool).
        profit_aware_operators: Use profit-weighted removal/insertion operators.
    """

    # Search Control
    max_iterations: int = 5000
    time_limit: float = 60.0
    seed: Optional[int] = None

    # RL Adaptation Rates (Section 3.2)
    reward_improvement: float = 1.0
    penalty_worsening: float = 1.0
    punishment_type: str = "RL1"  # "RL1" | "RL2" | "RL3"

    # Utility Management (Table 1: best maxUtilityValue = 40)
    utility_upper_bound: float = 40.0
    min_utility: float = 0.0
    initial_utility: float = 30.0  # = 0.75 × utility_upper_bound

    # Great Deluge: declining boundary (Fig 2, Step 18)
    # level(t) = quality_lb + (f0 - quality_lb) × (1 − t / T)
    quality_lb: float = 0.0

    # Problem flags
    vrpp: bool = True
    profit_aware_operators: bool = False

    # Injected Acceptance Criterion
    acceptance_criterion: IAcceptanceCriterion = field(default_factory=lambda: None)  # type: ignore

    @classmethod
    def from_config(cls, config: Any) -> "RLGDHHParams":
        """Create parameters from a configuration object."""
        # Build parameters
        params = cls(
            max_iterations=getattr(config, "max_iterations", 5000),
            time_limit=getattr(config, "time_limit", 60.0),
            reward_improvement=getattr(config, "reward_improvement", 1.0),
            penalty_worsening=getattr(config, "penalty_worsening", 1.0),
            punishment_type=getattr(config, "punishment_type", "RL1"),
            utility_upper_bound=getattr(config, "utility_upper_bound", 40.0),
            min_utility=getattr(config, "min_utility", 0.0),
            initial_utility=getattr(config, "initial_utility", 30.0),
            quality_lb=getattr(config, "quality_lb", 0.0),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )

        # Handle Acceptance Criterion Injection

        acceptance_cfg = getattr(config, "acceptance", None)
        if acceptance_cfg:
            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name=acceptance_cfg.method,
                config=acceptance_cfg.params,
            )
        else:
            # Default to great_deluge (gd) as it's the core of RL-GD-HH
            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name="gd",
                initial_level=params.quality_lb,  # quality_lb serves as the floor target
                decay_rate=0.01,
            )

        return params
