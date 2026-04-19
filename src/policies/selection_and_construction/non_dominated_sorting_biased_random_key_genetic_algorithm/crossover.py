"""
BRKGA Biased Uniform Crossover Module.

Implements the biased uniform crossover operator introduced in Gonçalves &
Resende (2011) for Biased Random-Key Genetic Algorithms.

In standard uniform crossover each gene is inherited with probability 0.5
from either parent.  BRKGA skews this probability to ``bias_elite`` (default
0.7) in favour of the elite parent, exploiting the fact that elite chromosomes
encode high-quality solutions.  The non-elite parent provides diversity.

References:
    Gonçalves, J. F., & Resende, M. G. (2011).
        Biased random-key genetic algorithms for combinatorial optimization.
        *Journal of Heuristics*, 17(5), 487–525.
"""

import numpy as np

from logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome import (
    Chromosome,
)


def biased_crossover(
    elite: Chromosome,
    non_elite: Chromosome,
    bias_elite: float,
    rng: np.random.Generator,
) -> Chromosome:
    """
    Produce one offspring via biased uniform crossover.

    For each gene position ``k``, the offspring inherits the value from
    *elite* with probability ``bias_elite``, and from *non_elite* with
    probability ``1 - bias_elite``.

    Args:
        elite: The elite parent chromosome (rank-1 or crowd-distance-preferred).
        non_elite: The non-elite parent chromosome (drawn from the non-elite pool).
        bias_elite: Probability of inheriting each gene from *elite*.
            Should be in ``(0.5, 1.0)`` to provide meaningful bias.
        rng: NumPy random Generator for reproducibility.

    Returns:
        A new :class:`~.chromosome.Chromosome` with ``n_bins`` from *elite*.

    Raises:
        ValueError: If the two parents have different ``n_bins``.
    """
    if elite.n_bins != non_elite.n_bins:
        raise ValueError(f"Parent n_bins mismatch: {elite.n_bins} vs {non_elite.n_bins}")

    n = len(elite.keys)
    mask = rng.uniform(0.0, 1.0, size=n) < bias_elite
    child_keys = np.where(mask, elite.keys, non_elite.keys)
    return Chromosome(child_keys, elite.n_bins)
