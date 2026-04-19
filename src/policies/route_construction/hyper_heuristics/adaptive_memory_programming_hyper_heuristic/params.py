"""
Configuration parameters for the Adaptive Memory Programming Hyper-Heuristic (AMP-HH).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class AMPHHParams:
    """
    Configuration parameters for AMP-HH.

    Attributes:
        mem_size: Size of the adaptive memory.
        iters: Number of iterations.
        seed: Random seed.
    """

    mem_size: int = 10
    iters: int = 50
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> AMPHHParams:
        """Create AMPHHParams from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            mem_size=getattr(config, "mem_size", 10),
            iters=getattr(config, "iters", 50),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert AMPParams to a dictionary for backend compatibility."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
