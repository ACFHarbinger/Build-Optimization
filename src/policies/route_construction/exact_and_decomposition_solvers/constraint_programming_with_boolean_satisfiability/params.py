"""
Parameter dataclasses for Constraint Programming (CP-SAT) solver.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CPSATParams:
    """
    Standardized parameters for the CP-SAT solver.
    """

    num_days: int = 3
    time_limit: float = 300.0
    search_workers: int = 8
    mip_gap: float = 0.01
    scaling_factor: int = 1000

    waste_weight: float = 1.0
    cost_weight: float = 1.0
    overflow_penalty: float = 10.0
    mean_increment: float = 0.2

    seed: Optional[int] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CPSATParams":
        """
        Create a CPSATParams instance from a raw configuration dictionary.

        Args:
            config: Dictionary containing parameter overrides.

        Returns:
            A CPSATParams instance with values mapped from the config.
        """
        return cls(
            num_days=int(config.get("num_days", 3)),
            time_limit=float(config.get("time_limit", 300.0)),
            search_workers=int(config.get("search_workers", 8)),
            mip_gap=float(config.get("mip_gap", 0.01)),
            scaling_factor=int(config.get("scaling_factor", 1000)),
            waste_weight=float(config.get("waste_weight", 1.0)),
            cost_weight=float(config.get("cost_weight", 1.0)),
            overflow_penalty=float(config.get("overflow_penalty", 10.0)),
            mean_increment=float(config.get("mean_increment", 0.2)),
            seed=config.get("seed"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameters to a dictionary.
        """
        return {
            "num_days": self.num_days,
            "time_limit": self.time_limit,
            "search_workers": self.search_workers,
            "mip_gap": self.mip_gap,
            "scaling_factor": self.scaling_factor,
            "waste_weight": self.waste_weight,
            "cost_weight": self.cost_weight,
            "overflow_penalty": self.overflow_penalty,
            "mean_increment": self.mean_increment,
            "seed": self.seed,
        }
