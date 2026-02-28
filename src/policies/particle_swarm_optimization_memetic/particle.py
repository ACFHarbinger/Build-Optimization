"""
Particle class for the Particle Swarm Optimization Memetic Algorithm (PSOMA).
"""

import numpy as np


class PSOMAParticle:
    """
    A single PSO particle representing a build solution.

    Attributes:
        build: Current item-slot allocation array.
        score: Objective value of current position.
        pbest_build: Personal best build.
        pbest_score: Objective value of personal best.
    """

    def __init__(self, build: np.ndarray, score: float):
        self.build = build.copy()
        self.score = score
        self.pbest_build = build.copy()
        self.pbest_score = score
