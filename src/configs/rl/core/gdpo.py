"""GDPO specific configuration."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GDPOConfig:
    """GDPO specific configuration."""

    objective_keys: List[str] = field(default_factory=lambda: ["reward_waste", "reward_cost"])
    objective_weights: Optional[List[float]] = None
    conditional_key: Optional[str] = None
    renormalize: bool = True
