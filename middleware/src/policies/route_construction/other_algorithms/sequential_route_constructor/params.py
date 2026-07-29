"""
Configuration parameters for the Sequential Route Constructor (SRC).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class SRCParams:
    """
    Configuration parameters for the Sequential Route Constructor.

    Attributes:
        constructors: List of route constructor names to run in sequence.
        time_limit: Maximum wall-clock time (seconds) for the entire sequence.
        seed: Random seed for deterministic execution.
    """

    constructors: List[str]
    time_limit: float = 60.0
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> SRCParams:
        """Create parameters from a configuration object."""
        return cls(
            constructors=getattr(config, "constructors", ["tsp", "nn"]),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", 42),
        )
