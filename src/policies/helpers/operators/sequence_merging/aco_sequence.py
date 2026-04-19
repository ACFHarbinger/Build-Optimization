"""
ACO Sequence Constructor Module.

Implements an Ant Colony Optimisation (ACO)-based hyper-heuristic that builds
dynamic operator execution sequences rather than routing tours.  Each "ant"
constructs a sequence of low-level heuristic (LLH) names by following
pheromone trails and heuristic desirability scores.

Instead of selecting nodes for a route, the ant selects the next operator
to apply to the current solution, treating the operator space as the
construction graph.

Pheromone update rule (ACS-style):
    τ_{ij}  ←  (1 − ρ) · τ_{ij}  +  Δτ_{ij}
    Δτ_{ij} =  Q / objective_delta    (if improvement occurred)

Transition probability:
    p(j | i) = [τ_{ij}^α · η_{ij}^β] / Σ_{k} [τ_{ik}^α · η_{ik}^β]

where η_{ij} is the heuristic desirability (historical average improvement
when applying operator j after i).

References:
    Dorigo, M., & Gambardella, L. M. (1997). Ant Colony System: A Cooperative
    Learning Approach to the Travelling Salesman Problem. IEEE TEVC, 1(1), 53–66.

    Burke, E. K. et al. (2013). Hyper-heuristics: A survey of the state of
    the art. Journal of the Operational Research Society, 64(12), 1695–1724.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.sequence_merging.aco_sequence import (
    ...     AcoSequenceState,
    ...     aco_build_sequence,
    ...     aco_update_pheromones,
    ... )
    >>> state = AcoSequenceState(op_names=["2opt", "or_opt", "double_bridge"])
    >>> seq = aco_build_sequence(state, seq_length=5)
    >>> aco_update_pheromones(state, seq, improvement=12.5)
"""

from random import Random
from typing import Dict, List, Optional

import numpy as np


class AcoSequenceState:
    """
    Persistent ACO pheromone and heuristic state for operator sequencing.

    Attributes:
        op_names: Ordered list of LLH operator names forming the construction graph.
        tau: Pheromone matrix (n_ops x n_ops).  tau[i][j] is the pheromone
            on the edge "apply operator j after operator i".
        eta: Heuristic desirability matrix (n_ops x n_ops).  Initialised to 1
            and updated with running-average improvement observations.
        alpha: Pheromone importance weight.
        beta: Heuristic importance weight.
        rho: Pheromone evaporation rate ∈ (0, 1].
        Q: Pheromone deposit constant.
        tau_min: Lower pheromone bound (prevents stagnation).
        tau_max: Upper pheromone bound (prevents dominance).
    """

    def __init__(
        self,
        op_names: List[str],
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        Q: float = 1.0,
        tau_min: float = 0.01,
        tau_max: float = 10.0,
        tau_init: float = 1.0,
    ) -> None:
        self.op_names = list(op_names)
        self.n = len(op_names)
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.tau_min = tau_min
        self.tau_max = tau_max
        self._idx: Dict[str, int] = {name: i for i, name in enumerate(op_names)}

        # Initialise pheromone and heuristic matrices
        self.tau: np.ndarray = np.full((self.n, self.n), tau_init, dtype=float)
        np.fill_diagonal(self.tau, 0.0)  # Disallow self-loops

        self.eta: np.ndarray = np.ones((self.n, self.n), dtype=float)
        np.fill_diagonal(self.eta, 0.0)

        # Running mean improvement for eta updates: (sum, count) per edge
        self._eta_sum: np.ndarray = np.ones((self.n, self.n), dtype=float)
        self._eta_cnt: np.ndarray = np.ones((self.n, self.n), dtype=float)

    def index(self, name: str) -> int:
        return self._idx[name]


def aco_build_sequence(
    state: AcoSequenceState,
    seq_length: int,
    start_op: Optional[str] = None,
    rng: Optional[Random] = None,
) -> List[str]:
    """
    Construct a single operator execution sequence using ACO transition rules.

    Each step selects the next operator probabilistically according to the
    combined pheromone–heuristic score.  An ε-greedy exploitation step is
    used: with probability 0.9 the highest-scoring successor is chosen
    (ACS exploitation), otherwise proportional roulette selection is used
    (ACS exploration).

    Args:
        state: Persistent ACO pheromone / heuristic state.
        seq_length: Number of operators in the output sequence (≥ 1).
        start_op: Name of the first operator in the sequence.  If None, the
            first operator is chosen uniformly at random.
        rng: Random number generator.

    Returns:
        List[str]: Sequence of operator names of length ``seq_length``.
    """
    if rng is None:
        rng = Random()

    if state.n == 0 or seq_length <= 0:
        return []

    current = rng.choice(state.op_names) if start_op is None else start_op

    sequence: List[str] = [current]
    for _ in range(seq_length - 1):
        current = _acs_transition(state, current, rng)
        sequence.append(current)

    return sequence


def aco_update_pheromones(
    state: AcoSequenceState,
    sequence: List[str],
    improvement: float,
    evaporate_all: bool = True,
) -> None:
    """
    Update pheromone trails based on the observed solution improvement.

    Applies global evaporation to all edges and deposits pheromone on the
    edges used in ``sequence`` proportional to ``improvement``.

    Args:
        state: Persistent ACO pheromone / heuristic state (mutated in-place).
        sequence: The operator sequence that was executed.
        improvement: Observed objective improvement (Δ cost or Δ profit).
            Negative values are treated as zero (no deposit for worsening).
        evaporate_all: If True, apply global evaporation before depositing.
    """
    if evaporate_all:
        state.tau *= 1.0 - state.rho
        state.tau = np.clip(state.tau, state.tau_min, state.tau_max)

    if improvement <= 0.0 or len(sequence) < 2:
        return

    deposit = state.Q / (improvement + 1e-9)

    for step in range(len(sequence) - 1):
        i = state.index(sequence[step])
        j = state.index(sequence[step + 1])
        state.tau[i, j] = min(state.tau[i, j] + deposit, state.tau_max)

        # Update heuristic desirability (running mean)
        state._eta_sum[i, j] += improvement
        state._eta_cnt[i, j] += 1
        state.eta[i, j] = state._eta_sum[i, j] / state._eta_cnt[i, j]


def aco_best_sequence(
    state: AcoSequenceState,
    seq_length: int,
    start_op: Optional[str] = None,
) -> List[str]:
    """
    Greedily construct the sequence by always choosing the highest-scoring next operator.

    This deterministic variant produces the pheromone-optimal sequence according
    to the current trail.  Useful for extracting the best-known policy after
    online ACO learning.

    Args:
        state: Persistent ACO pheromone / heuristic state.
        seq_length: Number of operators in the output sequence.
        start_op: Starting operator name; chosen by maximum column sum if None.

    Returns:
        List[str]: Greedy operator sequence.
    """
    if state.n == 0 or seq_length <= 0:
        return []

    if start_op is None:
        col_sums = (state.tau**state.alpha) * (state.eta**state.beta)
        best_start = int(np.argmax(col_sums.sum(axis=0)))
        current = state.op_names[best_start]
    else:
        current = start_op

    sequence: List[str] = [current]
    for _ in range(seq_length - 1):
        i = state.index(current)
        scores = (state.tau[i] ** state.alpha) * (state.eta[i] ** state.beta)
        scores[i] = 0.0  # no self-loop
        best_j = int(np.argmax(scores))
        current = state.op_names[best_j]
        sequence.append(current)

    return sequence


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _acs_transition(
    state: AcoSequenceState,
    current: str,
    rng: Random,
    exploit_prob: float = 0.9,
) -> str:
    """
    ACS-style transition: exploit (greedy) with probability exploit_prob,
    explore (proportional roulette) otherwise.
    """
    i = state.index(current)
    scores = (state.tau[i] ** state.alpha) * (state.eta[i] ** state.beta)
    scores[i] = 0.0  # Prevent self-loops

    total = float(scores.sum())
    if total < 1e-12:
        # All weights degenerate: uniform fallback
        candidates = [name for name in state.op_names if name != current]
        return rng.choice(candidates) if candidates else current

    if rng.random() < exploit_prob:
        return state.op_names[int(np.argmax(scores))]

    # Proportional roulette
    probs = scores / total
    cumulative = float(0.0)
    dart = rng.random()
    for j, p in enumerate(probs):
        cumulative += float(p)
        if dart <= cumulative:
            return state.op_names[j]

    return state.op_names[int(np.argmax(scores))]
