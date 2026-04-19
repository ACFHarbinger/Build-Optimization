"""
Parameters for GIHH (Hyper-Heuristic with Two Guidance Indicators).

This module defines the configuration parameters for the GIHH algorithm.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion

if TYPE_CHECKING:
    pass


@dataclass
class GIHHParams:
    """
    Configuration parameters for GIHH algorithm.

        # Episodic Learning parameters (Chen et al. 2018)
        seg (int): Segment size for episodic weight updates.
        alpha (float): Weight momentum factor.
        beta (float): Quality reward weight parameter.
        gamma (float): Directional reward penalty multiplier.
        min_prob (float): Minimum selection probability for any operator.

        # Stopping criteria
        nonimp_threshold (int): Maximum iterations without improvement (NONIMP).
    """

    # Core parameters
    time_limit: float = 60.0
    max_iterations: int = 1000
    seed: Optional[int] = None

    # Episodic Weight Updates
    seg: int = 80
    alpha: float = 0.5
    beta: float = 0.4
    gamma: float = 0.1
    min_prob: float = 0.05

    # Stopping criteria
    nonimp_threshold: int = 150

    # Profit-awareness
    vrpp: bool = True
    profit_aware_operators: bool = False
    acceptance_criterion: Optional[IAcceptanceCriterion] = None

    @classmethod
    def from_config(cls, config: Any) -> "GIHHParams":
        """Create parameters from a configuration object."""
        # Build parameters
        params = cls(
            time_limit=getattr(config, "time_limit", 60.0),
            max_iterations=getattr(config, "max_iterations", 1000),
            seed=getattr(config, "seed", None),
            seg=getattr(config, "seg", 80),
            alpha=getattr(config, "alpha", 0.5),
            beta=getattr(config, "beta", 0.4),
            gamma=getattr(config, "gamma", 0.1),
            min_prob=getattr(config, "min_prob", 0.05),
            nonimp_threshold=getattr(config, "nonimp_threshold", 150),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
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
            # Default to only_improving for standard GIHH
            params.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")

        return params
