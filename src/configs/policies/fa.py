"""
FA (Firefly Algorithm) configuration.
"""

from dataclasses import dataclass


@dataclass
class FAConfig:
    engine: str = "fa"
    pop_size: int = 20
    beta0: float = 1.0
    gamma: float = 0.1
    alpha_profit: float = 0.5
    beta_will: float = 0.3
    gamma_cost: float = 0.2
    alpha_rnd: float = 0.2
    max_iterations: int = 100
    n_removal: int = 3
    time_limit: float = 60.0
