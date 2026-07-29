"""
Configuration parameters for the Neural Policy.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class NeuralParams:
    """
    Configuration parameters for the Neural Policy.

    Attributes:
        waste_weight: Reward multiplier for collected waste (revenue).
        cost_weight: Penalty multiplier for travel distance.
        overflow_penalty: Penalty multiplier for bin overflows.
        selector_name: Name of vectorized selector for mandatory filtering.
        selector_threshold: Confidence threshold for node selection.
        seed: Random seed for reproducibility.
    """

    waste_weight: float = 1.0
    cost_weight: float = 1.0
    overflow_penalty: float = 1.0
    selector_name: Optional[str] = None
    selector_threshold: float = 0.7
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> NeuralParams:
        """Create NeuralParams from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            waste_weight=getattr(config, "waste_weight", 1.0),
            cost_weight=getattr(config, "cost_weight", 1.0),
            overflow_penalty=getattr(config, "overflow_penalty", 1.0),
            selector_name=getattr(config, "selector_name", None),
            selector_threshold=getattr(config, "selector_threshold", 0.7),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Params to a dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
