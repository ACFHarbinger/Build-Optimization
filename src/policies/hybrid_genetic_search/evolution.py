"""
Evolutionary operators for Hybrid Genetic Search (HGS).
"""

import random
from typing import List

import numpy as np

from core.problem import BuildProblem

from .individual import Individual


def evaluate(individual: Individual, problem: BuildProblem):
    """Evaluate an individual's score and feasibility."""
    individual.score = problem.evaluate(individual.build)
    individual.is_feasible = problem.is_feasible(individual.build)


def update_biased_fitness(population: List[Individual], elite_size: int):
    """Compute biased fitness based on score ranking and diversity contribution."""
    if not population:
        return

    # Sort by score descending
    population.sort(key=lambda x: x.score, reverse=True)
    for i, ind in enumerate(population):
        ind.rank = i

    # Diversity: average Hamming distance to k nearest neighbors
    for ind in population:
        dists = []
        for other in random.sample(population, min(len(population), 10)):
            if ind is other:
                continue
            d = np.sum(ind.build != other.build)
            dists.append(d)
        ind.diversity = np.mean(dists) if dists else 0.0

    # Combine rank and diversity (biased fitness: lower is better)
    max_div = max((ind.diversity for ind in population), default=1.0) + 1e-9
    for ind in population:
        # Fitness = rank + (1 - normalized_diversity)
        ind.fitness = ind.rank + elite_size * (1.0 - (ind.diversity / max_div))


def discrete_crossover(p1: Individual, p2: Individual) -> Individual:
    """Discrete crossover for build individuals."""
    child_build = p1.build.copy()
    mask = np.random.random(size=child_build.shape) < 0.5
    child_build[mask] = p2.build[mask]
    return Individual(child_build)
