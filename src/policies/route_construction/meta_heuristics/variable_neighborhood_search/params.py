"""
Configuration parameters for the Variable Neighborhood Search (VNS) solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion


@dataclass
class VNSParams:
    """
    Configuration for the VNS solver.

    Systematically explores a hierarchy of shaking neighborhoods (N_1 ... N_{k_max})
    with a local search descent between each shaking step.  An improvement resets
    k to 1; exhausting all k_max structures completes one outer iteration.

    Attributes:
        k_max: Number of shaking neighborhood structures (N_1 ... N_{k_max}).
        max_iterations: Total outer VNS iterations.
        local_search_iterations: LLH attempts per local search descent phase.
        n_removal: Nodes removed per LLH destroy step in local search.
        n_llh: Number of LLHs in the local search pool.
        time_limit: Wall-clock time limit in seconds.
        seed: Random seed for reproducibility.
        vrpp: Whether the problem is VRPP (True) or CVRP (False).
        profit_aware_operators: Whether to use profit-aware insertion/removal.
    """

    k_max: int = 5
    max_iterations: int = 200
    local_search_iterations: int = 20
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    # Injected Acceptance Criterion
    acceptance_criterion: IAcceptanceCriterion = field(default_factory=lambda: None)  # type: ignore

    @classmethod
    def from_config(cls, config: Any) -> VNSParams:
        """Build parameters from a configuration object."""
        # Build parameters
        params = cls(
            k_max=getattr(config, "k_max", 5),
            max_iterations=getattr(config, "max_iterations", 200),
            local_search_iterations=getattr(config, "local_search_iterations", 20),
            n_removal=getattr(config, "n_removal", 2),
            n_llh=getattr(config, "n_llh", 5),
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
            # Default to only_improving for standard VNS
            params.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")

        return params
