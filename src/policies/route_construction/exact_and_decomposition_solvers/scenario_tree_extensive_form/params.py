"""
Parameter dataclasses for Scenario-Tree Extensive Form (ST-EF) solver.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class STEFParams:
    """
    Standardized parameters for the ST-EF solver.
    """

    num_days: int = 3
    num_realizations: int = 3
    mean_increment: float = 0.2
    time_limit: float = 300.0
    mip_gap: float = 0.05
    waste_weight: float = 1.0
    cost_weight: float = 1.0
    overflow_penalty: float = 10.0
    discount_factor: float = 0.95
    use_mtz: bool = True
    seed: Optional[int] = 42

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "STEFParams":
        """
        Create a STEFParams instance from a raw configuration dictionary.

        Args:
            config: Dictionary containing parameter overrides.

        Returns:
            A STEFParams instance with values mapped from the config.
        """
        return cls(
            num_days=int(config.get("num_days", 3)),
            num_realizations=int(config.get("num_realizations", 3)),
            mean_increment=float(config.get("mean_increment", 0.2)),
            time_limit=float(config.get("time_limit", 300.0)),
            mip_gap=float(config.get("mip_gap", 0.05)),
            waste_weight=float(config.get("waste_weight", 1.0)),
            cost_weight=float(config.get("cost_weight", 1.0)),
            overflow_penalty=float(config.get("overflow_penalty", 10.0)),
            discount_factor=float(config.get("discount_factor", 0.95)),
            use_mtz=bool(config.get("use_mtz", True)),
            seed=config.get("seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameters to a dictionary.
        """
        return {
            "num_days": self.num_days,
            "num_realizations": self.num_realizations,
            "mean_increment": self.mean_increment,
            "time_limit": self.time_limit,
            "mip_gap": self.mip_gap,
            "waste_weight": self.waste_weight,
            "cost_weight": self.cost_weight,
            "overflow_penalty": self.overflow_penalty,
            "discount_factor": self.discount_factor,
            "use_mtz": self.use_mtz,
            "seed": self.seed,
        }
