"""
Sequence Merging Operators Package.

This package implements hyper-heuristic operators that build and evolve
*sequences of other algorithmic operators* rather than directly manipulating
routing solutions.  The operator sequences form dynamic execution chains that
are constructed, sampled, and recombined as first-class objects.

Paradigm: Operator Compounding
Objective: Hyper-Heuristic Control
Framework references: HH-ACO (Hyper-Heuristic Ant Colony Optimisation),
                      SS-HH (Sequential-based Selection Hyper-Heuristic)

Modules:
    aco_sequence
        ACO-based sequence construction.  Builds operator chains using
        pheromone trails and heuristic desirability scores in the operator
        transition graph.  Provides online pheromone learning via
        ``aco_update_pheromones`` and greedy extraction via
        ``aco_best_sequence``.

    markov_chain_sequence
        Markov Chain transition sampling.  Maintains a row-stochastic
        transition probability matrix T[i, j] = P(j | i) and samples operator
        sequences from it.  Supports EMA online updates, batch MLE fitting
        from historical execution logs, and stationary-distribution analysis.

    sequence_recombination
        Genetic operators for evolving operator sequences as chromosomes.
        Provides three crossover variants (single-point, uniform, OPX) and
        four mutation variants (substitution, insertion, deletion,
        transposition).

    sequential_selection
        Sequential Selection Hyper-Heuristic.  Maintains per-operator
        performance scores updated via additive, EMA, or sliding-window
        strategies.  Exposes ε-greedy, greedy, and Boltzmann (softmax)
        selection rules.  Can build full sequences or serve as a one-shot
        selector.

State objects (``AcoSequenceState``, ``MarkovSequenceState``, ``SsHhState``)
are mutable and designed to persist across search iterations for online
learning.

Example:
    >>> from logic.src.policies.helpers.operators.sequence_merging import (
    ...     AcoSequenceState,
    ...     aco_build_sequence,
    ...     aco_update_pheromones,
    ...     MarkovSequenceState,
    ...     markov_sample_sequence,
    ...     SsHhState,
    ...     ss_hh_select,
    ...     ss_hh_update,
    ...     sequence_single_point_crossover,
    ... )
"""

from .aco_sequence import (
    AcoSequenceState,
    aco_best_sequence,
    aco_build_sequence,
    aco_update_pheromones,
)
from .markov_chain_sequence import (
    MarkovSequenceState,
    markov_fit_from_log,
    markov_sample_sequence,
    markov_stationary_distribution,
    markov_update,
)
from .sequence_recombination import (
    sequence_deletion_mutation,
    sequence_insertion_mutation,
    sequence_order_preserving_crossover,
    sequence_single_point_crossover,
    sequence_substitution_mutation,
    sequence_transposition_mutation,
    sequence_uniform_crossover,
)
from .sequential_selection import (
    SsHhState,
    ss_hh_build_sequence,
    ss_hh_decay_scores,
    ss_hh_rank_operators,
    ss_hh_select,
    ss_hh_update,
)

__all__ = [
    # ACO sequence construction
    "AcoSequenceState",
    "aco_build_sequence",
    "aco_update_pheromones",
    "aco_best_sequence",
    # Markov chain sequence sampling
    "MarkovSequenceState",
    "markov_sample_sequence",
    "markov_update",
    "markov_fit_from_log",
    "markov_stationary_distribution",
    # Sequence crossover / mutation (recombination)
    "sequence_single_point_crossover",
    "sequence_uniform_crossover",
    "sequence_order_preserving_crossover",
    "sequence_substitution_mutation",
    "sequence_insertion_mutation",
    "sequence_deletion_mutation",
    "sequence_transposition_mutation",
    # Sequential selection
    "SsHhState",
    "ss_hh_select",
    "ss_hh_update",
    "ss_hh_build_sequence",
    "ss_hh_rank_operators",
    "ss_hh_decay_scores",
]
