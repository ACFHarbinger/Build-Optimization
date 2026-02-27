"""
QDE (Quantum-Inspired Differential Evolution) configuration.
"""

from dataclasses import dataclass


@dataclass
class QDEConfig:
    engine: str = "qde"
    pop_size: int = 20
    F: float = 0.5
    CR: float = 0.7
    max_iterations: int = 200
    n_removal: int = 2
    time_limit: float = 60.0
