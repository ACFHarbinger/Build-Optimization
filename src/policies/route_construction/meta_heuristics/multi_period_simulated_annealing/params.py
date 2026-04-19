"""
Configuration parameters for the Multi-Period Boltzmann-Metropolis Criterion (MP-BMC / Simulated Annealing).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class MP_BMC_Params:
    """
    Configuration parameters for MP-BMC.

    Attributes:
        iters: Number of iterations.
        init_temp: Initial temperature.
        cooling_rate: Cooling rate.
        seed: Random seed.
    """

    iters: int = 500
    init_temp: float = 100.0
    cooling_rate: float = 0.95
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> MP_BMC_Params:
        """Create MP_BMC_Params from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            iters=getattr(config, "iters", getattr(config, "max_iter", 500)),
            init_temp=getattr(config, "init_temp", 100.0),
            cooling_rate=getattr(config, "cooling_rate", 0.95),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert MP_BMC_Params to a dictionary for backend compatibility."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
