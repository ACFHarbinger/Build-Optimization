"""
ARCO (Adaptive Route Constructor Orchestrator) configuration dataclass.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ARCOConfig:
    """
    Configuration for the Adaptive Route Constructor Orchestrator (ARCO).

    Attributes:
        constructors: Ordered candidate list of route constructor names.
            The orchestrator selects a permutation of these at runtime.
        time_limit: Maximum wall-clock time (seconds) for the full chain.
        selection_strategy: How to select the next constructor.
            - ``"epsilon_greedy"``: exploit best weight with probability
              (1 − epsilon), explore uniformly otherwise.
            - ``"greedy"``: always pick the highest-weight constructor.
            - ``"softmax"``: Boltzmann-proportional sampling.
        epsilon: Exploration rate for ε-greedy strategy ∈ [0, 1].
        temperature: Softmax temperature > 0 (lower → greedier).
        alpha_ema: EMA smoothing factor for online weight updates ∈ (0, 1].
            Higher values weight recent observations more heavily.
        weight_init: Initial value for all pair-wise transition weights.
        weight_floor: Minimum allowed weight (prevents degenerate collapse).
        decay: Multiplicative weight decay applied each call ∈ (0, 1].
            Useful for non-stationary problems; 1.0 disables decay.
        seed: Random seed for reproducible exploration.
    """

    constructors: List[str] = field(default_factory=lambda: ["nn", "alns"])
    time_limit: float = 120.0
    selection_strategy: str = "epsilon_greedy"
    epsilon: float = 0.15
    temperature: float = 1.0
    alpha_ema: float = 0.15
    weight_init: float = 1.0
    weight_floor: float = 0.01
    decay: float = 1.0
    seed: int = 42
