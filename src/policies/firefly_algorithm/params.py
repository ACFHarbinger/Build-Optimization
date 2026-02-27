"""
Configuration parameters for the Discrete Firefly Algorithm (FA) solver.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FAParams:
    """
    Configuration parameters for the Discrete Firefly Algorithm.

    Firefly brightness = net profit.  Attractiveness decays with the
    discrete routing distance (swap distance between route sets).  A less
    bright firefly moves toward a brighter one via node-extraction and
    guided insertion using a favourability score.

    Attributes:
        pop_size: Number of fireflies.
        beta0: Maximum attractiveness (at distance 0).
        gamma: Light absorption coefficient controlling distance decay.
        alpha_profit: Weight of node profit in favourability score.
        beta_will: Weight of node willingness (waste fill fraction) in score.
        gamma_cost: Weight of insertion cost in favourability score (penalty).
        alpha_rnd: Probability of random-walk perturbation per firefly per iter.
        max_iterations: Maximum algorithm iterations.
        time_limit: Wall-clock time limit in seconds.
    """

    pop_size: int = 20
    beta0: float = 1.0
    gamma: float = 0.1
    alpha_profit: float = 0.5
    beta_will: float = 0.3
    gamma_cost: float = 0.2
    alpha_rnd: float = 0.2
    n_removal: int = 3
    max_iterations: int = 100
    time_limit: float = 60.0
