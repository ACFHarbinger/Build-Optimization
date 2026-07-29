"""
Configuration parameters for the Hierarchical Neural Agent (HNA).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class HNAParams:
    """
    Configuration parameters for HNA.

    Attributes:
        checkpoint_path: Path to a pre-trained HNAModule checkpoint.
        device: Torch device string.
        horizon: Lookahead horizon.
        overflow_penalty: Per-unit overflow penalty.
        greedy_threshold: Fill-level threshold (%) for fallback.
        seed: Random seed.
        verbose: Enable logging.
    """

    checkpoint_path: Optional[str] = None
    device: str = "cpu"
    horizon: int = 7
    overflow_penalty: float = 500.0
    greedy_threshold: float = 75.0
    seed: Optional[int] = None
    verbose: bool = False

    @classmethod
    def from_config(cls, config: Any) -> HNAParams:
        """Create HNAParams from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            checkpoint_path=getattr(config, "checkpoint_path", None),
            device=getattr(config, "device", "cpu"),
            horizon=getattr(config, "horizon", 7),
            overflow_penalty=getattr(config, "overflow_penalty", 500.0),
            greedy_threshold=getattr(config, "greedy_threshold", 75.0),
            seed=getattr(config, "seed", None),
            verbose=getattr(config, "verbose", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert HNAParams to a dictionary for backend compatibility."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
