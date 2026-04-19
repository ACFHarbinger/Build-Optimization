"""
Configuration dataclass for the HNA policy (simulator-callable wrapper config).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class HNAPolicyConfig:
    """Configuration for the Hierarchical Neural Agent simulator policy adapter.

    This is the lightweight config for the *simulator-callable* shim
    (``policy_hna.py``), not the full Lightning training module config.

    Attributes:
        checkpoint_path: Path to a pre-trained HNAModule checkpoint.
            If None, the manager falls back to a threshold-based heuristic
            and the worker uses the greedy nearest-neighbour heuristic.
        device: Torch device string ("cpu", "cuda", "cuda:0", etc.).
        horizon: Lookahead horizon passed to the manager observation.
        overflow_penalty: Per-unit overflow penalty used in reward computation.
        greedy_threshold: Fill-level threshold (%) for the fallback mandatory
            selector when no checkpoint is provided.
        seed: Random seed for stochastic components.
        verbose: Enable per-day inference logging.
    """

    checkpoint_path: Optional[str] = None
    device: str = "cpu"
    horizon: int = 7
    overflow_penalty: float = 500.0
    greedy_threshold: float = 75.0
    seed: Optional[int] = None
    verbose: bool = False
