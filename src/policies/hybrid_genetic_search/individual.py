"""
Individual representation for Hybrid Genetic Search (HGS).
"""

import numpy as np


class Individual:
    """
    Represents a candidate build in the HGS population.
    """

    def __init__(self, build: np.ndarray):
        self.build = build.copy()
        self.score = 0.0
        self.fitness = 0.0  # Biased fitness (score rank + diversity)
        self.diversity = 0.0
        self.rank = 0
        self.is_feasible = True

    def __repr__(self) -> str:
        return f"Individual(score={self.score:.2f}, fitness={self.fitness:.2f})"
