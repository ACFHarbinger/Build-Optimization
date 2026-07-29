"""
Configuration parameters for the Artificial Bee Colony (ABC) solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ABCParams:
    """
    Configuration parameters for the ABC solver.

    Mimics honey-bee foraging: each food source is a routing solution.
    Employed bees exploit their source; onlooker bees are directed by
    fitness-proportionate selection; scout bees reinitialise stagnant sources.

    Attributes:
        n_sources: Number of food sources (employed bees).
        limit: Trial counter threshold before a source is abandoned.
        max_iterations: Maximum ABC cycles.
        n_removal: Number of nodes removed per neighbourhood perturbation.
        local_search_iterations: Number of local search iterations.
        time_limit: Wall-clock time limit in seconds.
    """

    n_sources: int = 20
    limit: int = 10
    max_iterations: int = 200
    n_removal: int = 1
    local_search_iterations: int = 100
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "ABCParams":
        """Create parameters from a configuration object."""
        return cls(
            n_sources=max(1, getattr(config, "n_sources", 20)),
            limit=max(1, getattr(config, "limit", 10)),
            max_iterations=max(1, getattr(config, "max_iterations", 200)),
            n_removal=max(1, getattr(config, "n_removal", 1)),
            local_search_iterations=max(0, getattr(config, "local_search_iterations", 100)),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
