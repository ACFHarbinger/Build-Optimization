"""
Configuration parameters for the ALNS-IPO solver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params import ALNSParams


@dataclass
class ALNSIPOParams(ALNSParams):
    """
    Configuration parameters for the ALNS-IPO solver.

    Extends ALNSParams with multi-period stochastic routing parameters.

    Attributes:
        horizon: Planning horizon T (number of days).
        stockout_penalty: Penalty per unit of bin overflow.
        forward_looking_depth: Lookahead depth H for ForwardLookingInsertion.
        inter_period_operators: If False, disable inter-period operators.
        shift_direction: Direction for ShiftVisitRemoval ("both", "forward", "backward").
        inventory_lambda: Weight on inventory term in ForwardLookingInsertion.
        inter_period_weight: Initial roulette weight for inter-period operators.
        stochastic_repair: If True, multi-period repair ops use ScenarioTree evaluation.
    """

    horizon: int = 7
    stockout_penalty: float = 500.0
    forward_looking_depth: int = 3
    inter_period_operators: bool = True
    shift_direction: str = "both"
    inventory_lambda: float = 1.0
    inter_period_weight: float = 1.0
    stochastic_repair: bool = False

    @classmethod
    def from_config(cls, config: Any) -> ALNSIPOParams:
        """Create ALNSIPOParams from an ALNSIPOConfig dataclass or dict.

        Args:
            config: ALNSIPOConfig dataclass or dict with solver parameters.

        Returns:
            ALNSIPOParams instance with values from config.
        """
        # 1. Start with base ALNSParams initialization
        # We reuse the base class logic to handle standard ALNS fields and Acceptance Criterion injection
        base_params = super().from_config(config)
        base_dict = asdict(base_params)

        # 2. Extract IPO-specific fields
        ipo_fields = {
            "horizon": getattr(config, "horizon", 7),
            "stockout_penalty": getattr(config, "stockout_penalty", 500.0),
            "forward_looking_depth": getattr(config, "forward_looking_depth", 3),
            "inter_period_operators": getattr(config, "inter_period_operators", True),
            "shift_direction": getattr(config, "shift_direction", "both"),
            "inventory_lambda": getattr(config, "inventory_lambda", 1.0),
            "inter_period_weight": getattr(config, "inter_period_weight", 1.0),
            "stochastic_repair": getattr(config, "stochastic_repair", False),
        }

        # 3. Combine and instantiate
        # Note: We manually inject the acceptance_criterion since asdict() might not handle it if it's not a field
        # (though it is a field in ALNSParams)
        final_params = cls(**{**base_dict, **ipo_fields})
        final_params.acceptance_criterion = base_params.acceptance_criterion

        return final_params
