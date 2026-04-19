"""
ARCO (Adaptive Route Constructor Orchestrator) — Adaptive-Weight Meta-Constructor.

The ARCO is a meta-policy that coordinates multiple route constructors in a
dynamically ordered sequence.  Unlike the Sequential Route Constructor (SRC),
which executes constructors in a fixed, configuration-defined order, the ARCO
maintains two sets of adaptive weights that are updated online:

    w_first[i]      Desirability of constructor i as the *first* in the chain.
    W[i, j]         Conditional desirability of constructor j *immediately after*
                    constructor i, given i has already been placed in the chain.

At each call the ARCO builds a full permutation of its constructor pool by
sequentially eliminating already-selected constructors:

    Step 1 — First position:
        Select c_1 = argmax* w_first  (with exploration noise).

    Step 2 — Second position:
        Mask out c_1 from W[c_1, :], select c_2 from the remaining candidates.

    Step k — k-th position:
        Use W[c_{k-1}, :] as scores, mask already-chosen constructors, select c_k.

Each constructor in the resulting ordered chain receives the tour produced by
its predecessor as its warm start, giving ARCO the same state-threading
semantics as SRC.

After the full chain executes, ARCO observes per-step marginal improvement
(Δ profit or Δ cost) and updates W[c_{k-1}, c_k] via EMA:

    W[i, j] ← (1 − α) · W[i, j] + α · marginal_reward(j)

This allows the orchestrator to learn, across simulation days, which constructor
orderings are most effective for the current problem instance and distribution.

Selection strategies:
    - ``"epsilon_greedy"``: greedily follow weights with probability (1 − ε),
      explore uniformly with probability ε.
    - ``"greedy"``:         always pick the maximum-weight successor.
    - ``"softmax"``:        Boltzmann-proportional sampling; temperature τ
                            controls the sharpness of the distribution.

Registry key: ``"arco"``

References:
    Cowling, P., Kendall, G., & Soubeiga, E. (2001). A hyper-heuristic approach
    to scheduling a sales summit. PATAT III, LNCS 2079, 176–190.

    Burke, E. K., et al. (2013). Hyper-heuristics: A survey of the state of
    the art. Journal of the Operational Research Society, 64(12), 1695–1724.
"""

from __future__ import annotations

import logging
import time
from random import Random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from logic.src.interfaces.context.search_context import SearchContext

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.route_constructor import IRouteConstructor
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import ARCOParams

logger = logging.getLogger(__name__)


@GlobalRegistry.register(
    PolicyTag.HEURISTIC,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
    PolicyTag.ORCHESTRATOR,
    PolicyTag.ADAPTIVE_ALGORITHM,
)
@RouteConstructorRegistry.register("arco")
class AdaptiveRouteConstructorOrchestrator(BaseRoutingPolicy):
    """
    Adaptive Route Constructor Orchestrator (ARCO).

    A meta-constructor that maintains pair-wise transition weights between
    constructor names and uses them to dynamically order the execution chain
    at each call.  Weights are updated online from observed marginal
    improvements, allowing ARCO to learn which ordering performs best for the
    current problem instance.

    Architecture
    ------------
    Weight storage (persists across calls):
        w_first   : np.ndarray, shape (n,)       — first-position scores.
        W         : np.ndarray, shape (n, n)      — pair-wise transition matrix.
                    W[i, j] is the desirability of running constructor j
                    immediately after constructor i.  Diagonal is always 0.

    Selection at call time:
        1. Pick c_0  via  w_first  (using the configured selection strategy).
        2. Pick c_1  via  W[c_0, :]  after masking out c_0.
        3. Pick c_k  via  W[c_{k-1}, :]  after masking out {c_0, …, c_{k-1}}.

    Weight update after each call:
        For each consecutive pair (c_{k-1}, c_k) in the selected sequence:
            W[c_{k-1}, c_k] ← (1 − α) · W[c_{k-1}, c_k] + α · r_k
        where r_k is the normalised marginal reward from constructor c_k.
        The first-position weight is updated similarly:
            w_first[c_0] ← (1 − α) · w_first[c_0] + α · r_0

    Registry key: ``"arco"``
    """

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self.constructors: List[IRouteConstructor] = []
        self._constructor_names: List[str] = []
        self._initialized: bool = False

        # Build params from config
        if self.config is not None:
            self.params = ARCOParams.from_config(self.config)
        else:
            self.params = ARCOParams()

        # Adaptive weight state — initialised lazily once constructor names are known
        self._n: int = 0
        self._idx: Dict[str, int] = {}
        self._w_first: Optional[np.ndarray] = None  # shape (n,)
        self._W: Optional[np.ndarray] = None  # shape (n, n), diagonal = 0
        self._call_counts: Optional[np.ndarray] = None  # per-constructor call count
        self._rng: Random = Random(self.params.seed)

    # ------------------------------------------------------------------
    # BaseRoutingPolicy interface
    # ------------------------------------------------------------------

    @classmethod
    def _config_class(cls):
        from logic.src.configs.policies.arco import ARCOConfig

        return ARCOConfig

    @classmethod
    def _get_config_key(cls) -> str:
        return "arco"

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _initialize_constructors(self) -> None:
        """Lazily initialise sub-constructors and weight matrices."""
        if self._initialized:
            return

        from logic.src.policies.route_construction.base.factory import RouteConstructorFactory

        names: List[str] = self.params.constructors
        self.constructors = [RouteConstructorFactory.get_adapter(name) for name in names]
        self._constructor_names = list(names)
        self._n = len(names)
        self._idx = {name: i for i, name in enumerate(names)}
        self._init_weights()
        self._initialized = True

    def _init_weights(self) -> None:
        """Initialise weight matrices to uniform values."""
        n = self._n
        init = self.params.weight_init

        self._w_first = np.full(n, init, dtype=float)
        self._W = np.full((n, n), init, dtype=float)
        np.fill_diagonal(self._W, 0.0)  # Cannot follow yourself

        self._call_counts = np.zeros(n, dtype=int)

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def execute(
        self,
        **kwargs: Any,
    ) -> Tuple[
        Union[List[int], List[List[int]]],
        float,
        float,
        Optional[SearchContext],
        Optional[Any],
    ]:
        """
        Execute the adaptive constructor chain for one routing call.

        1. Uses current weights to order the constructor pool.
        2. Runs each constructor in sequence, threading state forward.
        3. Observes marginal improvement per step and updates weights.

        Args:
            **kwargs: Simulation context (tour, cost, profit, search_context,
                multi_day_context, mandatory, bins, distance_matrix, …).

        Returns:
            5-tuple: (tour, cost, profit, search_context, multi_day_context).
        """
        self._initialize_constructors()

        if not self.constructors:
            tour = kwargs.get("tour", [0, 0])
            cost = kwargs.get("cost", 0.0)
            profit = kwargs.get("profit", 0.0)
            return tour, cost, profit, kwargs.get("search_context"), kwargs.get("multi_day_context")

        # Apply per-call weight decay
        if self.params.decay < 1.0:
            self._apply_decay()

        start_time = time.perf_counter()

        # ---- Step 1: build the ordered sequence via adaptive selection ----
        sequence = self._select_sequence()
        logger.debug("ARCO selected sequence: %s", [self._constructor_names[i] for i in sequence])

        # ---- Step 2: execute the chain, tracking per-step performance ----
        current_tour: Union[List[int], List[List[int]]] = kwargs.get("tour", [0, 0])
        current_cost: float = kwargs.get("cost", 0.0)
        current_profit: float = kwargs.get("profit", 0.0)
        current_ctx: Optional[SearchContext] = kwargs.get("search_context")
        current_mdc: Optional[Any] = kwargs.get("multi_day_context")

        step_profits: List[float] = []
        step_costs: List[float] = []

        for step_idx, c_idx in enumerate(sequence):
            elapsed = time.perf_counter() - start_time
            if elapsed > self.params.time_limit:
                logger.warning(
                    "ARCO time limit reached (%.2fs > %.2fs). Stopping at step %d/%d.",
                    elapsed,
                    self.params.time_limit,
                    step_idx,
                    len(sequence),
                )
                break

            kwargs["tour"] = current_tour
            kwargs["cost"] = current_cost
            kwargs["profit"] = current_profit
            kwargs["search_context"] = current_ctx
            kwargs["multi_day_context"] = current_mdc

            prev_profit = current_profit
            prev_cost = current_cost

            constructor = self.constructors[c_idx]
            current_tour, current_cost, current_profit, current_ctx, current_mdc = constructor.execute(**kwargs)

            step_profits.append(current_profit - prev_profit)
            step_costs.append(prev_cost - current_cost)  # positive = improvement

        # ---- Step 3: online weight update ----
        self._update_weights(sequence[: len(step_profits)], step_profits, step_costs)

        return current_tour, current_cost, current_profit, current_ctx, current_mdc

    # ------------------------------------------------------------------
    # Adaptive sequence selection
    # ------------------------------------------------------------------

    def _select_sequence(self) -> List[int]:
        """
        Build a full permutation of constructor indices using adaptive weights.

        At step k:
          - k == 0: score vector = w_first.
          - k  > 0: score vector = W[prev, :].
          Mask out already-chosen constructors, then apply the configured
          selection strategy.

        Returns:
            List[int]: Constructor indices in the selected execution order.
        """
        assert self._w_first is not None
        assert self._W is not None

        chosen: List[int] = []
        available: List[int] = list(range(self._n))

        prev: Optional[int] = None

        while available:
            raw_scores = self._w_first.copy() if prev is None else self._W[prev, :].copy()

            # Mask already-chosen constructors
            masked_scores = np.full(self._n, -np.inf)
            for a in available:
                masked_scores[a] = max(raw_scores[a], self.params.weight_floor)

            next_idx = self._apply_strategy(masked_scores, available)
            chosen.append(next_idx)
            available.remove(next_idx)
            prev = next_idx

        return chosen

    def _apply_strategy(self, scores: np.ndarray, available: List[int]) -> int:
        """
        Apply the configured selection strategy over the available candidates.

        Args:
            scores: Full score vector (−inf for already-chosen).
            available: Indices of constructors still available.

        Returns:
            int: Selected constructor index.
        """
        strategy = self.params.selection_strategy

        if strategy == "greedy":
            return int(np.argmax(scores))

        if strategy == "epsilon_greedy":
            if self._rng.random() < self.params.epsilon:
                return self._rng.choice(available)
            return int(np.argmax(scores))

        if strategy == "softmax":
            avail_scores = np.array([scores[i] for i in available], dtype=float)
            avail_scores -= avail_scores.max()  # Numerical stability
            weights = np.exp(avail_scores / max(self.params.temperature, 1e-12))
            total = weights.sum()
            if total < 1e-12:
                return self._rng.choice(available)
            probs = weights / total
            dart = self._rng.random()
            cumulative = 0.0
            for j, p in enumerate(probs):
                cumulative += float(p)
                if dart <= cumulative:
                    return available[j]
            return available[int(np.argmax(probs))]

        raise ValueError(f"ARCO: unknown selection strategy {strategy!r}")

    # ------------------------------------------------------------------
    # Online weight update (EMA)
    # ------------------------------------------------------------------

    def _update_weights(
        self,
        sequence: List[int],
        step_profits: List[float],
        step_costs: List[float],
    ) -> None:
        """
        Update adaptive weights based on observed per-step marginal rewards.

        The reward signal combines profit gain and cost reduction, then is
        normalised to [0, 1] via a sigmoid-like clamp before the EMA step.
        Negative rewards (worsening steps) are clamped to 0 — the ARCO
        does not punish constructors that happen to worsen the interim
        objective; it only rewards those that improve it.

        Args:
            sequence: Constructor indices in the order they were executed.
            step_profits: Marginal profit change Δ for each step.
            step_costs: Marginal cost improvement Δ for each step (positive = better).
        """
        assert self._w_first is not None
        assert self._W is not None
        assert self._call_counts is not None

        alpha = self.params.alpha_ema
        floor = self.params.weight_floor

        for step, c_idx in enumerate(sequence):
            self._call_counts[c_idx] += 1

            # Combined reward: use profit if available, else cost
            raw_reward = step_profits[step] if step < len(step_profits) else 0.0
            if raw_reward <= 0.0 and step < len(step_costs):
                raw_reward = step_costs[step]

            # Sigmoid normalisation: maps any positive value to (0, 1)
            reward = _sigmoid_reward(raw_reward)

            if step == 0:
                # Update first-position weight
                self._w_first[c_idx] = max(
                    floor,
                    (1.0 - alpha) * self._w_first[c_idx] + alpha * reward,
                )
            else:
                prev_idx = sequence[step - 1]
                self._W[prev_idx, c_idx] = max(
                    floor,
                    (1.0 - alpha) * self._W[prev_idx, c_idx] + alpha * reward,
                )

    def _apply_decay(self) -> None:
        """Apply multiplicative decay to all weights, floored at weight_floor."""
        assert self._w_first is not None
        assert self._W is not None

        floor = self.params.weight_floor
        d = self.params.decay

        self._w_first = np.maximum(floor, self._w_first * d)
        self._W = np.maximum(floor, self._W * d)
        np.fill_diagonal(self._W, 0.0)  # Diagonal must stay 0

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_weight_summary(self) -> Dict[str, Any]:
        """
        Return a human-readable snapshot of current adaptive weights.

        Returns:
            Dict with keys:
                ``"first_position"``: dict mapping constructor name → w_first score.
                ``"transition_matrix"``: dict mapping (from, to) pair → W score.
                ``"call_counts"``: dict mapping constructor name → total call count.
        """
        self._initialize_constructors()
        assert self._w_first is not None
        assert self._W is not None
        assert self._call_counts is not None

        first_pos = {self._constructor_names[i]: float(self._w_first[i]) for i in range(self._n)}
        transitions = {
            (self._constructor_names[i], self._constructor_names[j]): float(self._W[i, j])
            for i in range(self._n)
            for j in range(self._n)
            if i != j
        }
        calls = {self._constructor_names[i]: int(self._call_counts[i]) for i in range(self._n)}
        return {
            "first_position": first_pos,
            "transition_matrix": transitions,
            "call_counts": calls,
        }

    def best_sequence(self) -> List[str]:
        """
        Return the greedy-optimal constructor sequence given current weights.

        Useful for logging or inspection after online learning has converged.

        Returns:
            List[str]: Constructor names in the optimal learned order.
        """
        self._initialize_constructors()
        assert self._w_first is not None
        assert self._W is not None

        # Temporarily override strategy with greedy for read-only extraction
        original_strategy = self.params.selection_strategy
        original_epsilon = self.params.epsilon
        self.params.selection_strategy = "greedy"
        self.params.epsilon = 0.0

        idx_seq = self._select_sequence()

        self.params.selection_strategy = original_strategy
        self.params.epsilon = original_epsilon

        return [self._constructor_names[i] for i in idx_seq]

    def reset_weights(self) -> None:
        """Reset all adaptive weights to their initial uniform values."""
        if self._initialized:
            self._init_weights()

    # ------------------------------------------------------------------
    # Required abstract method — not used (execute() is overridden)
    # ------------------------------------------------------------------

    def _run_solver(
        self,
        sub_dist_matrix: Any,
        sub_wastes: Any,
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Any,
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """Not used — execute() is overridden directly."""
        return [], 0.0, 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sigmoid_reward(raw: float, scale: float = 1.0) -> float:
    """
    Map a raw improvement value to a reward in [0, 1) via a sigmoid-like clamp.

    Non-positive values map to 0.  Large positive values approach 1.
    This keeps rewards bounded and comparable regardless of problem scale.

    f(x) = x / (x + scale)    for x > 0
    f(x) = 0                  for x ≤ 0
    """
    if raw <= 0.0:
        return 0.0
    return raw / (raw + scale)
