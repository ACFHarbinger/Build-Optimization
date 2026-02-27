"""
Configuration parameters for the Genetic Algorithm (GA) solver.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GAParams:
    """
    Configuration for the GA solver.

    Standard evolutionary algorithm with tournament selection, OX crossover,
    random relocate mutation, and elitism.

    Attributes:
        pop_size: Population size.
        max_generations: Number of evolutionary generations.
        crossover_rate: Probability of crossover between parents.
        mutation_rate: Probability of mutation per individual.
        tournament_size: Number of individuals in tournament selection.
        n_removal: Nodes removed per local search step.
        time_limit: Wall-clock time limit in seconds.
    """

    pop_size: int = 30
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    tournament_size: int = 3
    n_removal: int = 2
    time_limit: float = 60.0
