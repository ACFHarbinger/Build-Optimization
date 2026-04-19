"""
RL-GD-HH (Reinforcement Learning Great Deluge Hyper-Heuristic) configuration for Hydra.

This module defines the schema and default values for the RL-GD-HH policy,
aligned directly with the Ozcan et al. (2010) paper specification.

Paper mappings (Section 3, Table 1):
    - maxUtilityValue = utility_upper_bound = 40
    - Initial utility  = 0.75 × 40 = 30
    - qualityLB        = quality_lb = 0
    - Reward           = +1 (additive, RL1)
    - Penalty          = −1 (RL1), ÷2 (RL2), √ (RL3)
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from logic.src.configs.policies.other.acceptance_criteria import AcceptanceConfig


@dataclass
class RLGDHHConfig:
    """
    Configuration for the RL-GD-HH policy.

    The values defined here reproduce the best-performing configuration
    from Ozcan et al. (2010) unless otherwise noted.
    """

    # --- Search Control ---
    max_iterations: int = 5000
    time_limit: float = 60.0
    seed: Optional[int] = None

    # --- RL Adaptation Settings (Section 3.2) ---
    # Additive reward when a LLH strictly improves the solution (Step 14).
    reward_improvement: float = 1.0
    # Penalty magnitude applied for worsening AND neutral moves (Step 17).
    # The paper treats neutral moves identically to worsening ones.
    penalty_worsening: float = 1.0
    # Punishment variant: "RL1" (subtract), "RL2" (halve), "RL3" (root).
    punishment_type: str = "RL1"

    # --- Utility Constraints (Table 1: best maxUtilityValue = 40) ---
    utility_upper_bound: float = 40.0
    min_utility: float = 0.0
    # Initial utility = 0.75 × maxUtilityValue = 30 (per paper p. 16).
    initial_utility: float = 30.0

    # --- Great Deluge: Declining Boundary (Fig 2, Step 18) ---
    # level(t) = quality_lb + (f0 - quality_lb) × (1 − t / T)
    # qualityLB = 0 for all benchmark instances in the paper.
    quality_lb: float = 0.0

    # --- Problem Flags ---
    vrpp: bool = True

    # --- Acceptance Criterion ---
    # Default to Great Deluge to maintain parity with Ozcan et al. (2010)
    acceptance: AcceptanceConfig = field(
        default_factory=lambda: AcceptanceConfig(
            method="great_deluge",
            params={"quality_lb": 0.0},
        )
    )

    # --- Infrastructure Hooks ---
    mandatory_selection: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
