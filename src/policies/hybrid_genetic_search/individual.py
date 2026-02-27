"""
HGS Individual Module.

This module defines the Individual class used in the Hybrid Genetic Search (HGS)
algorithm. It represents a single solution in the population, storing genotype
(giant tour) and phenotype (routes), along with cost and fitness metrics.

Attributes:
    None

Example:
    >>> from logic.src.policies.hybrid_genetic_search.individual import Individual
    >>> ind = Individual(genotype=[1, 5, 2, ...], cost=100.0)
"""


class Individual:
    """
    Individual solution representation for genetic algorithm.
    """

    def __init__(self, giant_tour: list[int]):
        """
        Initialize an individual with a giant tour.

        Args:
            giant_tour: A list representing the visit order of all clients.
        """
        self.giant_tour = giant_tour
        self.routes: list[list[int]] = []
        self.fitness = -float("inf")
        self.profit_score = -float("inf")
        self.cost = 0.0
        self.revenue = 0.0

        self.dist_to_parents = 0.0
        self.rank_profit = 0
        self.rank_diversity = 0

    def __lt__(self, other: "Individual") -> bool:
        """Comparison based on biased fitness (used for sorting)."""
        return self.fitness < other.fitness
