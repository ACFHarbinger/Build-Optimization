"""
Configuration parameters for the Iterated Local Search (ILS) solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion


@dataclass
class ILSParams:
    """
    Configuration for the ILS solver.

    ILS alternates between a local search descent phase and a perturbation
    phase that escapes local optima.  Acceptance is based on strict profit
    improvement (hill-climbing) with the perturbation providing diversification.

    Attributes:
        n_restarts: Number of perturbation + descent cycles.
        inner_iterations: LLH iterations per descent phase.
        n_removal: Nodes removed per LLH destroy step.
        n_llh: Number of LLHs in the pool.
        perturbation_strength: Fraction of nodes perturbed.
        time_limit: Wall-clock time limit in seconds.
        seed: Random seed for reproducibility.
        vrpp: Whether the problem is VRPP (True) or CVRP (False).
        profit_aware_operators: Whether to use profit-aware insertion/removal.
    """

    n_restarts: int = 30
    inner_iterations: int = 20
    n_removal: int = 2
    n_llh: int = 5
    perturbation_strength: float = 0.15
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    # Injected Acceptance Criterion
    acceptance_criterion: IAcceptanceCriterion = field(default_factory=lambda: None)  # type: ignore

    @classmethod
    def from_config(cls, config: Any) -> ILSParams:
        """Build parameters from a configuration object."""
        # Build parameters
        params = cls(
            n_restarts=getattr(config, "n_restarts", 30),
            inner_iterations=getattr(config, "inner_iterations", 20),
            n_removal=getattr(config, "n_removal", 2),
            n_llh=getattr(config, "n_llh", 5),
            perturbation_strength=getattr(config, "perturbation_strength", 0.15),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
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
            # Default to only_improving for standard ILS
            params.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")

        return params
