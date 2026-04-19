"""
Configuration dataclass for the ADP Rollout policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ADPRolloutConfig:
    """Configuration for the Approximate Dynamic Programming (ADP) Rollout policy.

    Implements Powell's approximate DP framework for stochastic IRP using a
    truncated rollout with a fast greedy baseline to estimate V(S_{t+1}).

    Attributes:
        horizon: Planning horizon T (number of days).
        look_ahead_days: Rollout lookahead depth H (truncated horizon).
        n_scenarios: Number of scenarios sampled from the ScenarioTree.
        fill_threshold: Minimum fill level (%) to include a node as a candidate.
        candidate_strategy: Strategy for generating candidate node sets.
            ``"threshold"`` — include all nodes above ``fill_threshold``.
            ``"top_k"``     — include the top-K nodes by fill level.
            ``"beam"``      — explore ``max_candidate_sets`` random subsets.
        max_candidate_sets: Maximum number of candidate sets to evaluate (beam width).
        top_k: Number of top-fill nodes to consider when ``candidate_strategy="top_k"``.
        stockout_penalty: Penalty per unit of bin overflow.
        time_limit: Maximum wall-clock time per solve call (seconds).
        seed: Random seed for scenario sampling.
        verbose: Enable per-day logging.
    """

    horizon: int = 7
    look_ahead_days: int = 3
    n_scenarios: int = 10
    fill_threshold: float = 60.0
    candidate_strategy: str = "threshold"
    max_candidate_sets: int = 20
    top_k: int = 10
    stockout_penalty: float = 500.0
    time_limit: float = 60.0
    seed: Optional[int] = None
    verbose: bool = False
