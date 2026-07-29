"""
Configuration parameters for the Relaxation Enforced Neighborhood Search (RENS).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class RENSParams:
    """
    Configuration parameters for RENS.

    Attributes:
        time_limit: Total time budget for optimization.
        lp_time_limit: Time limit for the initial LP relaxation solve.
        mip_gap: Acceptable relative optimality gap.
        seed: Random seed.
    """

    time_limit: float = 300.0
    lp_time_limit: float = 60.0
    mip_gap: float = 0.01
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> RENSParams:
        """Create RENSParams from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            time_limit=getattr(config, "time_limit", 300.0),
            lp_time_limit=getattr(config, "lp_time_limit", 60.0),
            mip_gap=getattr(config, "mip_gap", 0.01),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert RENSParams to a dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
