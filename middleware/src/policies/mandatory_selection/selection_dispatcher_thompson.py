"""
Thompson Dispatcher Strategy Module.

Implements a contextual multi-armed bandit dispatcher that uses Thompson
Sampling to select the best "mandatory" strategy among a set of candidates
for the current operation. It maintains a Beta-Bernoulli posterior for each
candidate, which can be persisted across calls. High-performing strategies
(those resulting in lower operational cost/higher reward) are sampled
more frequently over time.

Note:
    This dispatcher uses a class-level shared state for tracking posteriors.
    In multi-tenant environments, instance-specific state should be used instead.

Example:
    >>> from logic.src.policies.helpers.mandatory.selection_dispatcher_thompson import ThompsonDispatcher
    >>> strategy = ThompsonDispatcher()
    >>> bins = strategy.select_bins(context)
"""

import logging
import os
import pickle
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import MandatorySelectionRegistry

# Configure simple logging
logger = logging.getLogger(__name__)


@MandatorySelectionRegistry.register("dispatcher_thompson")
class ThompsonDispatcher(IMandatorySelectionStrategy):
    """
    Contextual Thompson Sampling dispatcher for selection strategies.
    Hardened for thread-safety and principled exploration.
    """

    # Thread lock for state access
    _lock = threading.Lock()

    # Non-persistent class-level state for Beta distributions
    # key: strategy_name, value: (alpha, beta)
    _shared_state: Dict[str, Tuple[float, float]] = {}

    @classmethod
    def record_reward(
        cls,
        strategy_name: str,
        reward: float,
        baseline: Optional[float] = None,
        success_prob: Optional[float] = None,
        state_path: Optional[str] = None,
    ) -> None:
        """
        Update the posterior based on an observed reward.

        Args:
            strategy_name: Name of the strategy that was used.
            reward: Real-valued reward (e.g., negative cost).
            baseline: Optional baseline to binarize reward (success if reward > baseline).
            success_prob: Optional direct probability of success [0, 1]. If provided,
                          reward/baseline are ignored.
            state_path: Optional path to persist the updated state.
        """
        with cls._lock:
            # Load state
            state = cls._load_state_locked(state_path)

            alpha, beta = state.get(strategy_name, (1.0, 1.0))

            if success_prob is not None:
                p = np.clip(success_prob, 0.0, 1.0)
            elif baseline is not None:
                p = 1.0 if reward > baseline else 0.0
            else:
                # Default: no update or assume reward is already a probability/indicator
                # Using reward > 0 as a naive default but logging it.
                logger.warning("ThompsonDispatcher: No success_prob or baseline provided. Using reward > 0.")
                p = 1.0 if reward > 0 else 0.0

            # Fractional update for Beta-Bernoulli
            state[strategy_name] = (alpha + p, beta + (1.0 - p))

            # Save state
            cls._save_state_locked(state, state_path)

    @classmethod
    def _load_state_locked(cls, path: Optional[str]) -> Dict[str, Tuple[float, float]]:
        """Internal load within lock."""
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"ThompsonDispatcher: Failed to load state from {path}: {e}")
                return cls._shared_state
        return cls._shared_state

    @classmethod
    def _save_state_locked(cls, state: Dict[str, Tuple[float, float]], path: Optional[str]) -> None:
        """Internal save within lock."""
        cls._shared_state = state
        if path:
            dirname = os.path.dirname(path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            try:
                with open(path, "wb") as f:
                    pickle.dump(state, f)
            except Exception as e:
                logger.error(f"ThompsonDispatcher: Failed to save state to {path}: {e}")

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Sample a strategy via Thompson Sampling and dispatch the call.

        Args:
            context: SelectionContext with dispatcher configuration.

        Returns:
            List[int]: List of bin IDs (1-based index).
        """
        from logic.src.policies.mandatory_selection.base.selection_factory import MandatorySelectionFactory

        candidates = context.dispatcher_candidate_strategies or ["last_minute", "deadline", "mip_knapsack"]
        state_path = context.dispatcher_state_path
        temperature = max(context.dispatcher_exploration, 1e-9)

        with self._lock:
            state = self._load_state_locked(state_path)

        samples = {}
        for name in candidates:
            alpha, beta = state.get(name, (1.0, 1.0))

            # Principled temperature scaling: scale the deviation from the prior
            # alpha_prime = 1 + (alpha - 1) / T
            # T > 1 -> flatter (exploration)
            # T < 1 -> sharper (greedy)
            alpha_p = 1.0 + (alpha - 1.0) / temperature
            beta_p = 1.0 + (beta - 1.0) / temperature

            # Sample from Beta distribution
            theta = np.random.beta(alpha_p, beta_p)
            # Add tiny jitter for tie-breaking
            samples[name] = theta + 1e-12 * np.random.random()

        names = list(samples.keys())
        values = np.array([samples[n] for n in names])

        # Pick the winner using the highest sampled theta (with jitter)
        winner_idx = int(np.argmax(values))
        winner_name = names[winner_idx]

        # Dispatch
        strategy = MandatorySelectionFactory.create_strategy(winner_name)
        mandatory, sub_ctx = strategy.select_bins(context)

        # Merge metrics into a fresh context
        ctx = SearchContext.initialize(selection_metrics={"strategy": "ThompsonDispatcher", "winner": winner_name})
        return mandatory, ctx.merge(sub_ctx)
