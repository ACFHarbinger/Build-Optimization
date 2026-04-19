"""
Configuration parameters for Hybrid Genetic Search Adaptive Diversity Control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logic.src.configs.policies import HGSADCConfig


@dataclass
class HGSADCParams:
    pop_size: int = 25
    mu: int = 25
    nb_close: int = 4
    generations: int = 50
    n_vehicles: int = 0

    @classmethod
    def from_config(cls, config: "HGSADCConfig") -> "HGSADCParams":
        return cls(
            pop_size=getattr(config, "pop_size", 25),
            mu=getattr(config, "mu", 25),
            nb_close=getattr(config, "nb_close", 4),
            generations=getattr(config, "generations", 50),
            n_vehicles=getattr(config, "n_vehicles", 0),
        )
