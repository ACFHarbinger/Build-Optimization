"""
Bandit utilities for operator selection.
"""

from typing import Dict, List, Tuple

import numpy as np


class ThompsonBandit:
    """Thompson Sampling bandit using Beta-Bernoulli conjugate prior."""

    def __init__(self, names: List[str], seed: int = 42):
        self.names = names
        self.rng = np.random.default_rng(seed)
        # (alpha, beta) parameters
        self.params = {name: [1.0, 1.0] for name in names}

    def select(self) -> str:
        """Select an operator by sampling from Beta distributions."""
        samples: Dict[str, float] = {}
        for name, (a, b) in self.params.items():
            samples[name] = float(self.rng.beta(a, b))
        return max(samples, key=lambda x: samples[x])

    def update(self, name: str, success: int):
        """Update parameters based on success (1) or failure (0)."""
        if name in self.params:
            if success:
                self.params[name][0] += 1
            else:
                self.params[name][1] += 1

    def get_weights(self) -> Dict[str, Tuple[float, float]]:
        """Return the current alpha and beta parameters."""
        return {name: (a, b) for name, (a, b) in self.params.items()}

    def load_weights(self, weights: Dict[str, Tuple[float, float]]):
        """Load external weights."""
        for name, (a, b) in weights.items():
            if name in self.params:
                self.params[name] = [a, b]
