"""
KGLS algorithm parameters.
"""

from dataclasses import dataclass
from typing import List, Optional

from logic.src.configs.policies import KGLSConfig


@dataclass
class KGLSParams:
    """
    Runtime parameters for the KGLS solver.
    Extracts and manages configuration values during execution.
    """

    time_limit: float
    num_perturbations: int
    neighborhood_size: int
    moves: List[str]
    penalization_cycle: List[str]
    local_search_iterations: int
    seed: Optional[int]
    vrpp: bool = True
    profit_aware_operators: bool = True

    @classmethod
    def from_config(cls, config: KGLSConfig) -> "KGLSParams":
        """Build parameters from a KGLSConfig instance."""
        seed = getattr(config, "seed", None)
        vrpp = getattr(config, "vrpp", True)
        profit_aware_operators = getattr(config, "profit_aware_operators", False)

        return cls(
            time_limit=config.time_limit,
            num_perturbations=config.num_perturbations,
            neighborhood_size=config.neighborhood_size,
            moves=config.moves,
            penalization_cycle=config.penalization_cycle,
            local_search_iterations=config.local_search_iterations,
            seed=seed,
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
        )
