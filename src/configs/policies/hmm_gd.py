"""
HMM-GD (Hidden Markov Model - Great Deluge) configuration.
"""

from dataclasses import dataclass


@dataclass
class HMMGDConfig:
    engine: str = "hmm_gd"
    max_iterations: int = 500
    flood_margin: float = 0.05
    rain_speed: float = 0.001
    learning_rate: float = 0.1
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
