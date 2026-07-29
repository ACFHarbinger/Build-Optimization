"""
Configuration parameters for the ADP Rollout engine.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ADPRolloutParams:
    """Runtime parameters for the ADPRolloutEngine.

    Mirrors ``ADPRolloutConfig`` but is the object consumed by the engine
    at runtime (not serialised to YAML).

    Attributes:
        look_ahead_days: Rollout lookahead depth H.
        n_scenarios: Number of Monte Carlo scenarios per roll-out step.
        fill_threshold: Minimum fill level (%) to include a node as candidate.
        candidate_strategy: One of ``"threshold"``, ``"top_k"``, ``"beam"``.
        max_candidate_sets: Maximum number of candidate sets (beam strategy).
        top_k: Number of top-fill nodes for ``"top_k"`` strategy.
        stockout_penalty: Penalty per unit of overflow.
        time_limit: Maximum wall-clock time (seconds, 0 = unlimited).
        seed: Random seed.
        verbose: Enable per-day logging.
    """

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

    @classmethod
    def from_config(cls, config: Any) -> "ADPRolloutParams":
        """Create from an ADPRolloutConfig dataclass or dict.

        Args:
            config: ``ADPRolloutConfig`` dataclass or dict.

        Returns:
            ``ADPRolloutParams`` instance.
        """
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in dataclasses.fields(cls)}})
        return cls(
            look_ahead_days=getattr(config, "look_ahead_days", 3),
            n_scenarios=getattr(config, "n_scenarios", 10),
            fill_threshold=getattr(config, "fill_threshold", 60.0),
            candidate_strategy=getattr(config, "candidate_strategy", "threshold"),
            max_candidate_sets=getattr(config, "max_candidate_sets", 20),
            top_k=getattr(config, "top_k", 10),
            stockout_penalty=getattr(config, "stockout_penalty", 500.0),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            verbose=getattr(config, "verbose", False),
        )
