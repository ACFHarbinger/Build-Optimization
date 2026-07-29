from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SCAParams:
    """
    Configuration parameters for the SCA solver.

    Positions are updated using trigonometric sine/cosine functions.  A
    control parameter `a` decays from `a_max` to 0, shifting behaviour from
    global exploration (|sin/cos| > 1) to local exploitation (|sin/cos| < 1).
    The continuous position is binarised via sigmoid and decoded to a routing
    solution using the Largest Rank Value (LRV) rule.

    Attributes:
        pop_size: Population size.
        a_max: Initial value of the control parameter (decays to 0).
        max_iterations: Maximum SCA iterations.
        local_search_iterations: Maximum iterations for local search refinement.
        time_limit: Wall-clock time limit in seconds.
        vrpp: Whether to use VRPP expanded pool (True).
        profit_aware_operators: Whether to use profit-aware operators (True).
        seed: Random seed for reproducibility.
    """

    pop_size: int = 20
    a_max: float = 2.0
    max_iterations: int = 200
    local_search_iterations: int = 500
    time_limit: float = 60.0
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = None

    @classmethod
    def from_config(cls, config: Any) -> SCAParams:
        """Create SCAParams from a configuration object.

        Args:
            config: Configuration object or dictionary.

        Returns:
            SCAParams instance with values from config.
        """
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in dataclasses.fields(cls)}})

        return cls(
            pop_size=getattr(config, "pop_size", 20),
            a_max=getattr(config, "a_max", 2.0),
            max_iterations=getattr(config, "max_iterations", 200),
            local_search_iterations=getattr(config, "local_search_iterations", 500),
            time_limit=getattr(config, "time_limit", 60.0),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            seed=getattr(config, "seed", None),
        )
