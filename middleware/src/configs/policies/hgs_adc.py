from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HGSADCConfig:
    pop_size: int = 25
    mu: int = 25
    nb_close: int = 4
    generations: int = 50
    n_vehicles: int = 0
