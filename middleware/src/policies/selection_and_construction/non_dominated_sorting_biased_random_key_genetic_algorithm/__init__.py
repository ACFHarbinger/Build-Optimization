"""
NDS-BRKGA sub-package.

Exposes the public API of the Non-Dominated Sorting Biased Random-Key
Genetic Algorithm core:

    NDSBRKGAPolicy   — main policy (joint selection and construction)
    NDSBRKGAParams   — hyperparameter configuration
    Chromosome       — BRKGA chromosome with adaptive threshold decode
    Population       — population management (seeding, breeding, elite extraction)
"""

from .chromosome import Chromosome, compute_adaptive_thresholds
from .crossover import biased_crossover
from .nsga2 import crowding_distance, fast_non_dominated_sort, select_elite_nsga2
from .objectives import compute_overflow_risk, evaluate_chromosome, evaluate_population
from .params import NDSBRKGAParams
from .policy_nds_brkga import NDSBRKGAPolicy
from .population import Population

__all__ = [
    "Chromosome",
    "compute_adaptive_thresholds",
    "biased_crossover",
    "fast_non_dominated_sort",
    "crowding_distance",
    "select_elite_nsga2",
    "compute_overflow_risk",
    "evaluate_chromosome",
    "evaluate_population",
    "NDSBRKGAParams",
    "Population",
    "NDSBRKGAPolicy",
]
