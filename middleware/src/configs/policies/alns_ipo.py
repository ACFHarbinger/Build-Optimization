"""
Configuration dataclass for ALNS with Inter-Period Operators (ALNS-IPO).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ALNSIPOConfig:
    """Configuration for ALNS with Inter-Period Operators (IPO).

    Extends the standard ALNS with horizon-spanning destroy/repair operators that
    reason across the full T-day chromosome, enabling coordinated multi-day scheduling.

    Attributes:
        time_limit: Maximum runtime in seconds per solve call.
        max_iterations: Maximum ALNS iterations per solve call.
        start_temp: Initial temperature (0 = dynamic).
        cooling_rate: SA temperature decay per iteration.
        reaction_factor: Operator weight learning rate.
        min_removal: Minimum nodes to remove per destroy step.
        max_removal_pct: Maximum percentage of nodes to remove.
        max_removal_cap: Hard upper cap on removal count.
        segment_size: Iterations per operator-weight update segment.
        noise_factor: Noise level for repair operators.
        worst_removal_randomness: Randomness in worst removal (p ≥ 1).
        shaw_randomization: Shaw removal randomisation factor.
        regret_pool: Regret operator pool size ("regret2", "regret234", "regretAll").
        sigma_1: Score for new global best solution.
        sigma_2: Score for improving (but not best) solution.
        sigma_3: Score for accepted worse solution.
        vrpp: If True, allow inserting nodes not in the removed pool.
        profit_aware_operators: Use profit-aware operator variants.
        extended_operators: Add string/cluster/neighbor operators.
        seed: Random seed for reproducibility.
        engine: Engine backend ("custom").
        horizon: Planning horizon T (number of days).
        stockout_penalty: Penalty per unit of bin overflow.
        forward_looking_depth: Lookahead days H for insertion evaluation.
        inter_period_operators: Enable inter-period destroy operators.
        shift_direction: Direction of ShiftVisitRemoval ("both", "forward", "backward").
        inventory_lambda: Weight on inventory term in forward-looking insertion.
        inter_period_weight: Initial weight for inter-period operators.
    """

    # --- Standard ALNS parameters (mirrors ALNSParams) ---
    time_limit: float = 60.0
    max_iterations: int = 5000
    start_temp: float = 0.0
    cooling_rate: float = 0.995
    reaction_factor: float = 0.1
    min_removal: int = 4
    max_removal_pct: float = 0.3
    max_removal_cap: int = 100
    segment_size: int = 100
    noise_factor: float = 0.025
    worst_removal_randomness: float = 3.0
    shaw_randomization: float = 6.0
    regret_pool: str = "regret234"
    sigma_1: float = 33.0
    sigma_2: float = 9.0
    sigma_3: float = 13.0
    vrpp: bool = True
    profit_aware_operators: bool = False
    extended_operators: bool = False
    seed: Optional[int] = None
    engine: str = "custom"

    # --- Multi-period extensions ---
    horizon: int = 7
    stockout_penalty: float = 500.0
    forward_looking_depth: int = 3
    inter_period_operators: bool = True
    shift_direction: str = "both"
    inventory_lambda: float = 1.0
    inter_period_weight: float = 1.0
