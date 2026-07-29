"""
ILS-RVND-SP algorithm parameters.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion
from logic.src.policies.route_construction.acceptance_criteria.base.factory import AcceptanceCriterionFactory


@dataclass
class ILSRVNDSPParams:
    """
    Runtime parameters for the ILS-RVND-SP solver.
    Extracts and manages the configuration values during execution.
    """

    max_restarts: int = 10
    max_iter_ils: int = 100
    perturbation_strength: int = 2

    use_set_partitioning: bool = True
    mip_time_limit: float = 60.0
    sp_mip_gap: float = 0.01

    N: int = 150
    A: float = 11.0
    MaxIter_a: int = 50
    MaxIter_b: int = 100
    MaxIterILS_b: int = 2000
    TDev_a: float = 0.05
    TDev_b: float = 0.005
    max_vehicles: int = 0

    time_limit: float = 300.0
    seed: Optional[int] = 42
    vrpp: bool = True
    profit_aware_operators: bool = False
    local_search_iterations: int = 500

    engine: str = "gurobi"
    framework: str = "ortools"

    # Injected Acceptance Criterion
    acceptance_criterion: IAcceptanceCriterion = field(default_factory=lambda: None)  # type: ignore

    @classmethod
    def from_config(cls, config: Any) -> "ILSRVNDSPParams":
        """Build parameters from a configuration object."""
        # Build parameters
        params = cls(
            max_restarts=getattr(config, "max_restarts", 10),
            max_iter_ils=getattr(config, "max_iter_ils", 100),
            perturbation_strength=getattr(config, "perturbation_strength", 2),
            use_set_partitioning=getattr(config, "use_set_partitioning", True),
            mip_time_limit=getattr(config, "mip_time_limit", 60.0),
            sp_mip_gap=getattr(config, "sp_mip_gap", 0.01),
            N=getattr(config, "N", 150),
            A=getattr(config, "A", 11.0),
            MaxIter_a=getattr(config, "MaxIter_a", 50),
            MaxIter_b=getattr(config, "MaxIter_b", 100),
            MaxIterILS_b=getattr(config, "MaxIterILS_b", 2000),
            TDev_a=getattr(config, "TDev_a", 0.05),
            TDev_b=getattr(config, "TDev_b", 0.005),
            max_vehicles=getattr(config, "max_vehicles", 9999),
            time_limit=getattr(config, "time_limit", 300.0),
            seed=getattr(config, "seed", 42),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            local_search_iterations=getattr(config, "local_search_iterations", 500),
        )

        # Handle Acceptance Criterion Injection

        acceptance_cfg = getattr(config, "acceptance", None)
        if acceptance_cfg:
            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name=acceptance_cfg.method,
                config=acceptance_cfg.params,
            )
        else:
            # Default to only_improving for standard ILS-RVND-SP
            params.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")

        return params
