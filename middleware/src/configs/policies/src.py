"""
SRC (Sequential Route Constructor) configuration dataclass.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SRCConfig:
    """Configuration for the Sequential Route Constructor (SRC).

    Attributes:
        constructors: List of route constructor names to run in sequence.
        time_limit: Maximum wall-clock time (seconds) for the entire sequence.
    """

    constructors: List[str] = field(default_factory=lambda: ["tsp", "nn"])
    time_limit: float = 60.0
    seed: int = 42
