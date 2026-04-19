"""
Configuration parameters for the Reactive Tabu Search (RTS) solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion

if TYPE_CHECKING:
    from logic.src.configs.policies import RTSConfig


@dataclass
class RTSParams:
    """
    Configuration for the Reactive Tabu Search solver.

    RTS uses short-term memory (tabu list) to forbid recent moves and
    hash-based cycle detection to adaptively adjust tabu tenure.

    Attributes:
        initial_tenure: Starting tabu tenure.
        min_tenure: Minimum allowed tenure.
        max_tenure: Maximum allowed tenure.
        tenure_increase: Multiplicative factor on cycle detection.
        tenure_decrease: Multiplicative factor on long non-cycling periods.
        max_iterations: Total LLH applications.
        n_removal: Nodes removed per destroy step.
        n_llh: Number of LLHs in the pool.
        time_limit: Wall-clock time limit in seconds.
    """

    initial_tenure: int = 7
    min_tenure: int = 3
    max_tenure: int = 20
    tenure_increase: float = 1.5
    tenure_decrease: float = 0.9
    max_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False
    acceptance_criterion: Optional[IAcceptanceCriterion] = None

    @classmethod
    def from_config(cls, config: RTSConfig) -> RTSParams:
        """Create RTSParams from a Hydra configuration object."""
        params = cls(
            initial_tenure=getattr(config, "initial_tenure", 7),
            min_tenure=getattr(config, "min_tenure", 3),
            max_tenure=getattr(config, "max_tenure", 20),
            tenure_increase=getattr(config, "tenure_increase", 1.5),
            tenure_decrease=getattr(config, "tenure_decrease", 0.9),
            max_iterations=getattr(config, "max_iterations", 500),
            n_removal=getattr(config, "n_removal", 2),
            n_llh=getattr(config, "n_llh", 5),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )

        # Handle Acceptance Criterion Injection (Aspiration folding)
        from logic.src.policies.route_construction.acceptance_criteria.base.factory import AcceptanceCriterionFactory

        acceptance_cfg = getattr(config, "acceptance", None)
        if acceptance_cfg:
            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name=acceptance_cfg.method,
                config=acceptance_cfg.params,
            )
        else:
            # Default to aspiration for standard RTS
            params.acceptance_criterion = AcceptanceCriterionFactory.create(name="ac")

        return params
