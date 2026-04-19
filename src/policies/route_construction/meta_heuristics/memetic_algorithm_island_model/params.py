"""
Configuration parameters for Memetic Island Model Genetic Algorithm.

This matches the "Soccer League Competition (SLC)" algorithm structure
but uses rigorous Genetic Algorithm terminology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MemeticAlgorithmIslandModelParams:
    """
    Parameters for Memetic Algorithm with Island Model (MA-IM).

    Matches the Soccer League Competition (SLC) logic:
    - n_islands (n_teams): Number of sub-populations.
    - island_size (team_size): Individuals per sub-population.
    - max_generations (max_iterations): Evolution cycles.
    - n_removal: Nodes removed per mutation/perturbation step.
    - stagnation_limit: Generations without improvement before island regeneration.
    - local_search_iterations: Number of local search iterations.
    - time_limit: Wall-clock time limit in seconds.

    Reference:
        Moosavian, N., & Rppdsarou, B. K. (2014).
        "Soccer league competition algorithm: A novel meta-heuristic
        algorithm for optimal design of water distribution networks."
        Whitley, D., Rana, S., & Heckendorn, R. B. (1998).
        "The island model genetic algorithm: On separability, population size and convergence."
    """

    n_islands: int = 5  # K islands (analogous to teams)
    island_size: int = 4  # N individuals per island
    max_generations: int = 50
    stagnation_limit: int = 5
    n_removal: int = 1
    local_search_iterations: int = 100
    time_limit: float = 60.0
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = None

    @classmethod
    def from_config(cls, config: Any) -> "MemeticAlgorithmIslandModelParams":
        """Create parameters from a configuration object."""
        return cls(
            n_islands=getattr(config, "n_islands", 5),
            island_size=getattr(config, "island_size", 4),
            max_generations=getattr(config, "max_generations", 50),
            stagnation_limit=getattr(config, "stagnation_limit", 5),
            n_removal=getattr(config, "n_removal", 1),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 60.0),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            seed=getattr(config, "seed", 42),
        )
