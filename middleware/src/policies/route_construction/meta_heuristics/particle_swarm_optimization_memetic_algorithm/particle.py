"""
Particle class for the Particle Swarm Optimization Memetic Algorithm (PSOMA).
"""

import copy
from typing import List


class PSOMAParticle:
    """
    A single PSO particle representing a routing solution.

    Attributes:
        routes: Current route set.
        profit: Objective value of current position.
        pbest_routes: Personal best route set.
        pbest_profit: Objective value of personal best.
    """

    def __init__(self, routes: List[List[int]], profit: float):
        self.routes = routes
        self.profit = profit
        self.pbest_routes = copy.deepcopy(routes)
        self.pbest_profit = profit
