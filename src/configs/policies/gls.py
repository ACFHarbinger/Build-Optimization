"""
GLS (Guided Local Search) configuration.
"""

from dataclasses import dataclass


@dataclass
class GLSConfig:
    engine: str = "gls"
    lambda_param: float = 0.3
    max_restarts: int = 50
    n_removal: int = 2
    n_llh: int = 5
    inner_iterations: int = 20
    time_limit: float = 60.0
