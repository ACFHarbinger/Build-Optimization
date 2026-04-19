"""
Markov Chain Sequence Sampler Module.

Implements a Markov Chain–based hyper-heuristic that samples operator execution
sequences from a learned transition probability matrix T, where T[i][j] is the
probability of applying operator j immediately after operator i.

The transition matrix can be:
  - Initialised uniformly (equal probability for all successors).
  - Updated online via exponential moving average (EMA) based on observed
    quality improvements.
  - Fitted to historical execution logs via maximum-likelihood estimation.

This allows the system to learn which operator orderings tend to produce
good improvements and to sample those orderings preferentially during search.

References:
    Burke, E. K., et al. (2010). Classification of metaheuristics and design
    of experiments for the analysis of components. Annals of OR.

    Soria-Alcaraz, J. A., et al. (2014). Evolver: A desktop tool for the
    hyper-heuristic optimisation of combinatorial problems. Scientific
    Programming.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.sequence_merging.markov_chain_sequence import (
    ...     MarkovSequenceState,
    ...     markov_sample_sequence,
    ...     markov_update,
    ... )
    >>> state = MarkovSequenceState(op_names=["2opt", "or_opt", "relocate"])
    >>> seq = markov_sample_sequence(state, seq_length=4)
    >>> markov_update(state, seq, improvement=8.3)
"""

from random import Random
from typing import Dict, List, Optional

import numpy as np


class MarkovSequenceState:
    """
    Persistent Markov Chain transition state for operator sequencing.

    Attributes:
        op_names: Ordered list of LLH operator names.
        T: Row-stochastic transition probability matrix (n_ops x n_ops).
            T[i, j] = P(apply j | last applied i).
        alpha_ema: EMA smoothing factor for online updates ∈ (0, 1).
            Higher values give more weight to recent observations.
        visit_counts: Cumulative edge visit counts for diagnostics.
    """

    def __init__(
        self,
        op_names: List[str],
        alpha_ema: float = 0.1,
        allow_self_loops: bool = False,
    ) -> None:
        self.op_names = list(op_names)
        self.n = len(op_names)
        self.alpha_ema = alpha_ema
        self.allow_self_loops = allow_self_loops
        self._idx: Dict[str, int] = {name: i for i, name in enumerate(op_names)}

        # Uniform initialisation (excluding self-loops unless allowed)
        self.T: np.ndarray = np.ones((self.n, self.n), dtype=float)
        if not allow_self_loops:
            np.fill_diagonal(self.T, 0.0)
        self._normalise()

        self.visit_counts: np.ndarray = np.zeros((self.n, self.n), dtype=float)

    def _normalise(self) -> None:
        row_sums = self.T.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        self.T = self.T / row_sums

    def index(self, name: str) -> int:
        return self._idx[name]


def markov_sample_sequence(
    state: MarkovSequenceState,
    seq_length: int,
    start_op: Optional[str] = None,
    rng: Optional[Random] = None,
) -> List[str]:
    """
    Sample an operator execution sequence from the Markov transition matrix.

    Starting from ``start_op`` (or a uniformly random operator if None), each
    successor is drawn from the corresponding row of T.

    Args:
        state: Persistent Markov chain state.
        seq_length: Number of operators in the output sequence.
        start_op: Name of the first operator.  If None, chosen uniformly.
        rng: Random number generator.

    Returns:
        List[str]: Sampled sequence of operator names.
    """
    if rng is None:
        rng = Random()

    if state.n == 0 or seq_length <= 0:
        return []

    current = rng.choice(state.op_names) if start_op is None else start_op

    sequence: List[str] = [current]
    for _ in range(seq_length - 1):
        i = state.index(current)
        row = state.T[i]
        # Multinomial draw from row probabilities
        dart = rng.random()
        cumulative = 0.0
        next_op = state.op_names[-1]  # Fallback
        for j, p in enumerate(row):
            cumulative += float(p)
            if dart <= cumulative:
                next_op = state.op_names[j]
                break
        current = next_op
        sequence.append(current)

    return sequence


def markov_update(
    state: MarkovSequenceState,
    sequence: List[str],
    improvement: float,
    reward_threshold: float = 0.0,
) -> None:
    """
    Update the Markov transition matrix based on observed improvement.

    Uses EMA reinforcement: transitions that produced improvement receive
    higher probability weight.  Non-improving transitions are not penalised
    (neutral update), preserving exploratory capacity.

    Args:
        state: Persistent Markov chain state (mutated in-place).
        sequence: The operator sequence that was executed.
        improvement: Observed objective improvement.  Must be ≥ 0.
            Values ≤ reward_threshold are treated as non-improving.
        reward_threshold: Minimum improvement to trigger a positive update.
    """
    if len(sequence) < 2:
        return

    for step in range(len(sequence) - 1):
        i = state.index(sequence[step])
        j = state.index(sequence[step + 1])
        state.visit_counts[i, j] += 1

        if improvement > reward_threshold:
            # EMA boost: pull T[i,j] toward 1 scaled by normalised improvement
            reward = min(1.0, improvement / (improvement + 1.0))  # Sigmoid-like clamp
            state.T[i, j] = (1.0 - state.alpha_ema) * state.T[i, j] + state.alpha_ema * reward

    state._normalise()


def markov_fit_from_log(
    state: MarkovSequenceState,
    execution_log: List[List[str]],
) -> None:
    """
    Fit the transition matrix to a set of historical execution sequences via MLE.

    Counts all consecutive (op_i, op_j) transitions across the log and
    normalises each row.  Previous state is reset.

    Args:
        state: Persistent Markov chain state (mutated in-place).
        execution_log: List of previously executed operator sequences.
    """
    counts = np.zeros((state.n, state.n), dtype=float)
    if not state.allow_self_loops:
        np.fill_diagonal(counts, 0.0)

    for seq in execution_log:
        for step in range(len(seq) - 1):
            if seq[step] in state._idx and seq[step + 1] in state._idx:
                i = state.index(seq[step])
                j = state.index(seq[step + 1])
                if not state.allow_self_loops and i == j:
                    continue
                counts[i, j] += 1

    # Rows with zero counts keep uniform distribution
    row_sums = counts.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).flatten()
    counts[zero_rows] = 1.0
    if not state.allow_self_loops:
        for k in np.where(zero_rows)[0]:
            counts[k, k] = 0.0

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    state.T = counts / row_sums
    state.visit_counts = counts.copy()


def markov_stationary_distribution(state: MarkovSequenceState) -> np.ndarray:
    """
    Compute the stationary distribution π of the Markov chain.

    π is the normalised left eigenvector of T corresponding to eigenvalue 1,
    giving the long-run fraction of time spent at each operator.

    Args:
        state: Persistent Markov chain state.

    Returns:
        np.ndarray: Stationary probability vector of shape (n_ops,).
    """
    # Power iteration
    pi = np.ones(state.n, dtype=float) / state.n
    for _ in range(1000):
        pi_new = pi @ state.T
        if float(np.max(np.abs(pi_new - pi))) < 1e-9:
            break
        pi = pi_new
    return pi / pi.sum()
