"""
Configuration parameters for the Relax-and-Fix-and-Optimize (RFO) matheuristic.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class RFOParams:
    """
    Configuration parameters for Relax-and-Fix-and-Optimize.

    Attributes:
        window_size: Size of the rolling integer window.
        step_size: Number of days to slide the window forward after each solve.
        mip_time: Time limit for each sub-MIP solve in seconds.
        mip_gap: Target optimality gap for sub-MIP solves.
    """

    window_size: int = 3
    step_size: int = 2
    mip_time: float = 60.0
    mip_gap: float = 0.01

    @classmethod
    def from_config(cls, config: Any) -> RFOParams:
        """Create RFOParams from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            window_size=getattr(config, "window_size", 3),
            step_size=getattr(config, "step_size", 2),
            mip_time=getattr(config, "mip_time", 60.0),
            mip_gap=getattr(config, "mip_gap", 0.01),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert RFOParams to a dictionary for backend compatibility."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
