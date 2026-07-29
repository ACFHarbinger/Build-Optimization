"""
Evolutionary Mutation Operators Package.

This package implements post-crossover mutation operators for genetic algorithms
applied to VRP solutions. These operators apply localized, stochastic changes to
offspring chromosomes to maintain population diversity and prevent premature
convergence to poor local optima.

Paradigm: Stochastic Variation
Objective: Genetic Diversity
Framework references: GA (Genetic Algorithms), DE (Differential Evolution)

Operators:
    - swap_mutation / swap_mutation_profit
        Exchange two randomly chosen non-depot nodes in the flat chromosome.
        Analogue of bit-flip mutation for discrete permutation spaces.

    - inversion_mutation / inversion_mutation_profit
        Reverse a random contiguous segment within a single route.
        Equivalent to an unguided, non-improving 2-opt move.

    - scramble_mutation / scramble_mutation_profit
        Shuffle a random contiguous segment within a single route.
        Introduces higher positional disorder than inversion.

    - random_2opt_mutation / random_2opt_mutation_profit
        Apply a 2-opt reversal at random cut points without improvement check.
        Pure stochastic 2-opt; may worsen the current tour intentionally.

    - de_rand_1_mutation
        DE/rand/1 adapted for permutation chromosomes.  Operates on a
        population; generates one mutant per individual using three random donors.

    - de_best_1_mutation
        DE/best/1 variant: uses the population-best as the base vector.
        Faster convergence than DE/rand/1 at the cost of diversity.

Example:
    >>> from logic.src.policies.helpers.operators.evolutionary_mutation import (
    ...     swap_mutation,
    ...     inversion_mutation,
    ...     scramble_mutation,
    ...     random_2opt_mutation,
    ...     de_rand_1_mutation,
    ... )
"""

from .differential_evolution import de_best_1_mutation, de_rand_1_mutation
from .inversion import inversion_mutation, inversion_mutation_profit
from .random_2opt import random_2opt_mutation, random_2opt_mutation_profit
from .scramble import scramble_mutation, scramble_mutation_profit
from .swap import swap_mutation, swap_mutation_profit

EVOLUTIONARY_MUTATION_OPERATORS = {
    "SWAP": swap_mutation,
    "INVERSION": inversion_mutation,
    "SCRAMBLE": scramble_mutation,
    "RANDOM_2OPT": random_2opt_mutation,
    "DE_RAND_1": de_rand_1_mutation,
    "DE_BEST_1": de_best_1_mutation,
}

EVOLUTIONARY_MUTATION_NAMES = list(EVOLUTIONARY_MUTATION_OPERATORS.keys())

__all__ = [
    "EVOLUTIONARY_MUTATION_NAMES",
    "EVOLUTIONARY_MUTATION_OPERATORS",
    "swap_mutation",
    "swap_mutation_profit",
    "inversion_mutation",
    "inversion_mutation_profit",
    "scramble_mutation",
    "scramble_mutation_profit",
    "random_2opt_mutation",
    "random_2opt_mutation_profit",
    "de_rand_1_mutation",
    "de_best_1_mutation",
]
