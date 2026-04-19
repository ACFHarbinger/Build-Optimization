"""
Configuration parameters for the Simulated Annealing Neighborhood Search (SANS) policy.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class SANSParams:
    """
    Configuration parameters for the SANS solver.

    Attributes:
        engine: Optimization engine ('new' or 'og').
        T_init: Initial temperature for Simulated Annealing.
        iterations_per_T: Iterations before cooling.
        alpha: Cooling rate (geometric decay).
        T_min: Minimum temperature to stop search.
        time_limit: Total time limit for optimization.
        perc_bins_can_overflow: Fractional allowance for bin overflow during search.
        V: Volume param (original LAC engine).
        shift_duration: Duration of collection shift in seconds.
        combination: Strategy for LAC combination.
        seed: Random seed for reproducibility.
    """

    engine: str = "new"
    T_init: float = 75.0
    iterations_per_T: int = 5000
    alpha: float = 0.95
    T_min: float = 0.01
    time_limit: float = 60.0
    perc_bins_can_overflow: float = 0.0
    V: float = 0.0
    shift_duration: float = 28800.0  # 8 hours
    combination: str = "best"
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> SANSParams:
        """Create SANSParams from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            engine=getattr(config, "engine", "new"),
            T_init=getattr(config, "T_init", 75.0),
            iterations_per_T=getattr(config, "iterations_per_T", 5000),
            alpha=getattr(config, "alpha", 0.95),
            T_min=getattr(config, "T_min", 0.01),
            time_limit=getattr(config, "time_limit", 60.0),
            perc_bins_can_overflow=getattr(config, "perc_bins_can_overflow", 0.0),
            V=getattr(config, "V", 0.0),
            shift_duration=getattr(config, "shift_duration", 28800.0),
            combination=getattr(config, "combination", "best"),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Params to a dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
