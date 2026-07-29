"""
Configuration parameters for the Selection Hyper-Heuristic (SHH).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class SHHParams:
    """
    Configuration parameters for SHH.

    Attributes:
        iters: Number of iterations.
        history_len: Length of the Late Acceptance history.
        seed: Random seed.
    """

    iters: int = 200
    history_len: int = 10
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> SHHParams:
        """Create SHHParams from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            iters=getattr(config, "iters", 200),
            history_len=getattr(config, "history_len", 10),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert SHHParams to a dictionary for backend compatibility."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
