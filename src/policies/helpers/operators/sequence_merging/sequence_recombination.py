"""
Sequence Recombination Module.

Implements crossover and mutation operators that evolve *sequences of operators*
rather than routing solutions.  This is the recombination layer of a
meta-evolutionary hyper-heuristic: operator sequences are treated as
chromosomes, and standard GA operations (crossover, mutation) are applied to
produce new hybrid execution chains.

Sequence Crossover:
    Combines two parent operator sequences using:
    - Single-point crossover (SPX): cut each parent at one random point,
      swap the tails.
    - Uniform crossover (UX): at each position independently, take the gene
      from parent 1 with probability 0.5, otherwise from parent 2.
    - Order-preserving crossover (OPX): preserves the relative order of
      operators as they appear in parent 1, filling gaps with parent 2.

Sequence Mutation:
    - Substitution: replace a random operator with a randomly chosen one.
    - Insertion: insert a random operator at a random position.
    - Deletion: remove a random operator (minimum sequence length enforced).
    - Transposition: swap two randomly chosen operators in the sequence.

References:
    Burke, E. K., et al. (2013). Hyper-heuristics: A survey of the state of
    the art. Journal of the Operational Research Society, 64(12), 1695–1724.

    Cowling, P., Kendall, G., & Soubeiga, E. (2001). A hyper-heuristic approach
    to scheduling a sales summit. Proceedings of Practice and Theory of
    Automated Timetabling III, LNCS 2079.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.sequence_merging.sequence_recombination import (
    ...     sequence_single_point_crossover,
    ...     sequence_uniform_crossover,
    ...     sequence_substitution_mutation,
    ... )
    >>> child = sequence_single_point_crossover(["2opt", "or_opt", "kick"], ["relocate", "2opt"], rng=rng)
    >>> mutated = sequence_substitution_mutation(child, op_pool=["2opt", "or_opt", "kick", "relocate"])
"""

from random import Random
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Sequence Crossover operators
# ---------------------------------------------------------------------------


def sequence_single_point_crossover(
    parent1: List[str],
    parent2: List[str],
    rng: Optional[Random] = None,
) -> Tuple[List[str], List[str]]:
    """
    Single-point crossover between two operator sequences.

    Selects one cut point in each parent and exchanges the suffix segments to
    produce two children.  Child lengths vary (they inherit different-length
    tails from each parent).

    Args:
        parent1: First parent operator sequence.
        parent2: Second parent operator sequence.
        rng: Random number generator.

    Returns:
        Tuple[List[str], List[str]]: Two child operator sequences.
    """
    if rng is None:
        rng = Random()

    if not parent1 or not parent2:
        return list(parent1), list(parent2)

    c1 = rng.randint(0, len(parent1))
    c2 = rng.randint(0, len(parent2))

    child1 = parent1[:c1] + parent2[c2:]
    child2 = parent2[:c2] + parent1[c1:]

    return child1, child2


def sequence_uniform_crossover(
    parent1: List[str],
    parent2: List[str],
    swap_prob: float = 0.5,
    rng: Optional[Random] = None,
) -> Tuple[List[str], List[str]]:
    """
    Uniform crossover between two operator sequences.

    At each position up to max(len(parent1), len(parent2)), the gene is
    taken from parent1 with probability ``swap_prob`` and from parent2
    otherwise.  Positions beyond the shorter parent inherit from the longer.

    Args:
        parent1: First parent operator sequence.
        parent2: Second parent operator sequence.
        swap_prob: Probability of inheriting gene from parent1 at each step.
        rng: Random number generator.

    Returns:
        Tuple[List[str], List[str]]: Two child operator sequences.
    """
    if rng is None:
        rng = Random()

    n = max(len(parent1), len(parent2))
    p1 = parent1 + parent1[::-1] * (n // max(len(parent1), 1))  # cycle-extend if shorter
    p2 = parent2 + parent2[::-1] * (n // max(len(parent2), 1))

    # Safer: pad with repeated last element
    p1 = list(parent1) + [parent1[-1]] * max(0, n - len(parent1)) if parent1 else []
    p2 = list(parent2) + [parent2[-1]] * max(0, n - len(parent2)) if parent2 else []

    child1: List[str] = []
    child2: List[str] = []

    for i in range(n):
        if i < len(p1) and i < len(p2):
            if rng.random() < swap_prob:
                child1.append(p1[i])
                child2.append(p2[i])
            else:
                child1.append(p2[i])
                child2.append(p1[i])
        elif i < len(p1):
            child1.append(p1[i])
        elif i < len(p2):
            child2.append(p2[i])

    return child1 or list(parent1), child2 or list(parent2)


def sequence_order_preserving_crossover(
    parent1: List[str],
    parent2: List[str],
    rng: Optional[Random] = None,
) -> List[str]:
    """
    Order-preserving crossover (OPX) for operator sequences.

    Selects a random contiguous segment from parent1 and preserves it in the
    child at its original positions.  Remaining positions are filled with
    operators from parent2 in the order they appear, skipping duplicates
    only if the sequences are intended to have unique operators.

    Since operator sequences allow repetition (the same operator can appear
    multiple times), this implementation preserves all operators from parent2
    without uniqueness filtering.

    Args:
        parent1: First parent operator sequence.
        parent2: Second parent operator sequence.
        rng: Random number generator.

    Returns:
        List[str]: One child operator sequence.
    """
    if rng is None:
        rng = Random()

    if not parent1:
        return list(parent2)
    if not parent2:
        return list(parent1)

    n = len(parent1)
    c1, c2 = sorted(rng.sample(range(n + 1), 2))

    child: List[Optional[str]] = [None] * n  # type: ignore[assignment]
    child[c1:c2] = parent1[c1:c2]

    # Fill remaining positions from parent2 in order
    fill = list(parent2)
    fill_idx = 0
    for pos in range(n):
        if child[pos] is None:
            if fill_idx < len(fill):
                child[pos] = fill[fill_idx]
                fill_idx += 1
            else:
                child[pos] = parent1[pos]  # Fallback to parent1

    return [op for op in child if op is not None]


# ---------------------------------------------------------------------------
# Sequence Mutation operators
# ---------------------------------------------------------------------------


def sequence_substitution_mutation(
    sequence: List[str],
    op_pool: List[str],
    n_mutations: int = 1,
    rng: Optional[Random] = None,
) -> List[str]:
    """
    Substitution mutation: replace random operator(s) with alternatives from the pool.

    Args:
        sequence: Input operator sequence.
        op_pool: Full pool of available operator names to draw replacements from.
        n_mutations: Number of substitution events to apply.
        rng: Random number generator.

    Returns:
        List[str]: Mutated sequence (new list; original unchanged).
    """
    if rng is None:
        rng = Random()

    if not sequence or not op_pool:
        return list(sequence)

    result = list(sequence)
    for _ in range(n_mutations):
        pos = rng.randint(0, len(result) - 1)
        result[pos] = rng.choice(op_pool)

    return result


def sequence_insertion_mutation(
    sequence: List[str],
    op_pool: List[str],
    max_length: int = 20,
    rng: Optional[Random] = None,
) -> List[str]:
    """
    Insertion mutation: insert a randomly chosen operator at a random position.

    Args:
        sequence: Input operator sequence.
        op_pool: Full pool of available operator names.
        max_length: Maximum allowed sequence length after insertion.
        rng: Random number generator.

    Returns:
        List[str]: Mutated sequence with one operator inserted (if max_length
            permits).
    """
    if rng is None:
        rng = Random()

    if not op_pool or len(sequence) >= max_length:
        return list(sequence)

    result = list(sequence)
    pos = rng.randint(0, len(result))
    result.insert(pos, rng.choice(op_pool))
    return result


def sequence_deletion_mutation(
    sequence: List[str],
    min_length: int = 1,
    rng: Optional[Random] = None,
) -> List[str]:
    """
    Deletion mutation: remove a randomly chosen operator from the sequence.

    Args:
        sequence: Input operator sequence.
        min_length: Minimum sequence length; deletion is skipped if the
            sequence would fall below this threshold.
        rng: Random number generator.

    Returns:
        List[str]: Mutated sequence with one operator removed.
    """
    if rng is None:
        rng = Random()

    if len(sequence) <= min_length:
        return list(sequence)

    result = list(sequence)
    pos = rng.randint(0, len(result) - 1)
    result.pop(pos)
    return result


def sequence_transposition_mutation(
    sequence: List[str],
    rng: Optional[Random] = None,
) -> List[str]:
    """
    Transposition mutation: swap two randomly chosen operators in the sequence.

    Args:
        sequence: Input operator sequence.
        rng: Random number generator.

    Returns:
        List[str]: Mutated sequence with two positions exchanged.
    """
    if rng is None:
        rng = Random()

    if len(sequence) < 2:
        return list(sequence)

    result = list(sequence)
    i, j = rng.sample(range(len(result)), 2)
    result[i], result[j] = result[j], result[i]
    return result
