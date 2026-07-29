"""
Sequential Selection Hyper-Heuristic (SS-HH) Module.

Implements the Sequential Selection Hyper-Heuristic framework for operator
selection and sequencing.  SS-HH maintains a performance score for each
low-level heuristic (LLH) and selects the next operator to apply according
to one of three selection strategies:

  - Greedy (ε = 0): always pick the highest-scoring operator.
  - ε-Greedy: exploit the best with probability (1 − ε), explore randomly
    with probability ε.
  - Softmax (Boltzmann): sample proportional to exp(score / temperature),
    giving a smooth trade-off between exploration and exploitation.

Scores are updated after each operator application using one of:
  - Additive reinforcement: score += improvement.
  - Exponential Moving Average (EMA): score ← (1 − α) · score + α · reward.
  - Sliding-window average: maintain the last W observations and use their mean.

The operator returns a scored selection state object that persists across
iterations, enabling online learning of which operators are most effective
for the current problem instance.

References:
    Cowling, P., Kendall, G., & Soubeiga, E. (2001). A hyper-heuristic approach
    to scheduling a sales summit.  PATAT III, LNCS 2079, 176–190.

    Burke, E. K., Hyde, M., Kendall, G., & Woodward, J. (2012). A classification
    of hyper-heuristic approaches: Revisited.  In: Handbook of Metaheuristics.
    Springer, pp. 449–477.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.sequence_merging.ss_hh import (
    ...     SsHhState,
    ...     ss_hh_select,
    ...     ss_hh_update,
    ... )
    >>> state = SsHhState(op_names=["2opt", "or_opt", "relocate", "double_bridge"])
    >>> op_name = ss_hh_select(state, strategy="epsilon_greedy", epsilon=0.2)
    >>> ss_hh_update(state, op_name, improvement=5.3)
"""

from random import Random
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Update strategies
# ---------------------------------------------------------------------------
_UPDATE_ADDITIVE = "additive"
_UPDATE_EMA = "ema"
_UPDATE_SLIDING = "sliding"

# Selection strategies
_SELECT_GREEDY = "greedy"
_SELECT_EPSILON_GREEDY = "epsilon_greedy"
_SELECT_SOFTMAX = "softmax"


class SsHhState:
    """
    Persistent SS-HH operator scoring state.

    Attributes:
        op_names: Ordered list of LLH operator names.
        scores: Current performance scores for each operator.
        call_counts: Number of times each operator has been selected.
        improvement_totals: Cumulative improvement observed for each operator.
        _history: Per-operator sliding windows of recent improvements.
        alpha_ema: EMA smoothing factor (used when update_strategy == "ema").
        window_size: Sliding-window width (used when update_strategy == "sliding").
        update_strategy: One of "additive", "ema", "sliding".
        score_floor: Minimum allowed score (prevents score collapse).
    """

    def __init__(
        self,
        op_names: List[str],
        alpha_ema: float = 0.2,
        window_size: int = 10,
        update_strategy: str = _UPDATE_EMA,
        initial_score: float = 1.0,
        score_floor: float = 0.01,
    ) -> None:
        self.op_names: List[str] = list(op_names)
        self.n: int = len(op_names)
        self.alpha_ema = alpha_ema
        self.window_size = window_size
        self.update_strategy = update_strategy
        self.score_floor = score_floor
        self._idx: Dict[str, int] = {name: i for i, name in enumerate(op_names)}

        self.scores: np.ndarray = np.full(self.n, initial_score, dtype=float)
        self.call_counts: np.ndarray = np.zeros(self.n, dtype=int)
        self.improvement_totals: np.ndarray = np.zeros(self.n, dtype=float)
        # Sliding window history: list of recent improvement values per operator
        self._history: List[List[float]] = [[] for _ in range(self.n)]

    def index(self, name: str) -> int:
        return self._idx[name]

    def reset_scores(self, initial_score: float = 1.0) -> None:
        """Reset all scores and history to their initial state."""
        self.scores[:] = initial_score
        self.call_counts[:] = 0
        self.improvement_totals[:] = 0.0
        self._history = [[] for _ in range(self.n)]


def ss_hh_select(
    state: SsHhState,
    strategy: str = _SELECT_EPSILON_GREEDY,
    epsilon: float = 0.1,
    temperature: float = 1.0,
    rng: Optional[Random] = None,
) -> str:
    """
    Select the next operator according to the SS-HH selection strategy.

    Args:
        state: Persistent SS-HH scoring state.
        strategy: Selection strategy — one of:
            - ``"greedy"``: always pick the highest-score operator.
            - ``"epsilon_greedy"``: best with prob (1 − ε), random with prob ε.
            - ``"softmax"``: Boltzmann-proportional sampling.
        epsilon: Exploration probability for ε-greedy strategy.
        temperature: Softmax temperature τ > 0.  Lower values make selection
            more greedy; higher values approach uniform random.
        rng: Random number generator.

    Returns:
        str: Name of the selected operator.
    """
    if rng is None:
        rng = Random()

    if state.n == 0:
        raise ValueError("SsHhState has no operators registered.")

    if strategy == _SELECT_GREEDY:
        return state.op_names[int(np.argmax(state.scores))]

    if strategy == _SELECT_EPSILON_GREEDY:
        if rng.random() < epsilon:
            return rng.choice(state.op_names)
        return state.op_names[int(np.argmax(state.scores))]

    if strategy == _SELECT_SOFTMAX:
        # Boltzmann / softmax sampling
        logits = state.scores / max(temperature, 1e-12)
        logits -= logits.max()  # Numerical stability
        weights = np.exp(logits)
        total = weights.sum()
        if total < 1e-12:
            return rng.choice(state.op_names)
        probs = weights / total
        dart = rng.random()
        cumulative = 0.0
        for j, p in enumerate(probs):
            cumulative += float(p)
            if dart <= cumulative:
                return state.op_names[j]
        return state.op_names[int(np.argmax(probs))]

    raise ValueError(f"Unknown selection strategy: {strategy!r}")


def ss_hh_update(
    state: SsHhState,
    op_name: str,
    improvement: float,
) -> None:
    """
    Update the score of an operator based on the observed improvement.

    Args:
        state: Persistent SS-HH scoring state (mutated in-place).
        op_name: Name of the operator that was applied.
        improvement: Observed objective change.  Negative values represent
            worsening; the raw delta is passed to allow penalty schemes.
    """
    i = state.index(op_name)
    state.call_counts[i] += 1
    reward = max(0.0, improvement)  # Non-negative reward for standard SS-HH
    state.improvement_totals[i] += reward

    if state.update_strategy == _UPDATE_ADDITIVE:
        state.scores[i] = max(state.score_floor, state.scores[i] + reward)

    elif state.update_strategy == _UPDATE_EMA:
        state.scores[i] = max(
            state.score_floor,
            (1.0 - state.alpha_ema) * state.scores[i] + state.alpha_ema * reward,
        )

    elif state.update_strategy == _UPDATE_SLIDING:
        window = state._history[i]
        window.append(reward)
        if len(window) > state.window_size:
            window.pop(0)
        state.scores[i] = max(state.score_floor, float(np.mean(window)))

    else:
        raise ValueError(f"Unknown update strategy: {state.update_strategy!r}")


def ss_hh_build_sequence(
    state: SsHhState,
    seq_length: int,
    strategy: str = _SELECT_EPSILON_GREEDY,
    epsilon: float = 0.1,
    temperature: float = 1.0,
    rng: Optional[Random] = None,
) -> List[str]:
    """
    Build a complete operator execution sequence using repeated SS-HH selection.

    Each call to ``ss_hh_select`` chooses the next operator independently
    based on current scores, without any sequential state between positions
    (i.e., scores are independent of the preceding operator).

    Args:
        state: Persistent SS-HH scoring state.
        seq_length: Number of operators in the output sequence.
        strategy: Selection strategy (see ``ss_hh_select``).
        epsilon: Exploration probability for ε-greedy.
        temperature: Softmax temperature.
        rng: Random number generator.

    Returns:
        List[str]: Selected operator sequence.
    """
    if rng is None:
        rng = Random()

    return [
        ss_hh_select(state, strategy=strategy, epsilon=epsilon, temperature=temperature, rng=rng)
        for _ in range(seq_length)
    ]


def ss_hh_rank_operators(state: SsHhState) -> List[tuple]:
    """
    Return operators ranked by descending current score.

    Args:
        state: Persistent SS-HH scoring state.

    Returns:
        List[Tuple[str, float, int]]: Sorted list of (op_name, score, call_count)
            triples, best operator first.
    """
    ranked = sorted(
        zip(state.op_names, state.scores.tolist(), state.call_counts.tolist(), strict=False),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked


def ss_hh_decay_scores(
    state: SsHhState,
    decay: float = 0.99,
) -> None:
    """
    Apply multiplicative score decay to all operators.

    Useful for non-stationary problems where operator effectiveness changes
    over time.  Decayed scores are floored at ``state.score_floor``.

    Args:
        state: Persistent SS-HH scoring state (mutated in-place).
        decay: Multiplicative decay factor ∈ (0, 1].
    """
    state.scores = np.maximum(state.score_floor, state.scores * decay)
