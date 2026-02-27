"""
OBA (Old Bachelor Acceptance) configuration.
"""

from dataclasses import dataclass


@dataclass
class OBAConfig:
    engine: str = "oba"
    dilation: float = 5.0
    contraction: float = 2.0
    max_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
