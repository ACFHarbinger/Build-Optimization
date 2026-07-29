"""
Joint Simulated Annealing Parameters.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class JointSAParams:
    """
    Configuration parameters for the Joint Simulated Annealing solver.
    """

    # --- Annealing Schedule ---
    start_temp: float = 1000.0
    cooling_rate: float = 0.995
    max_steps: int = 2000
    restart_limit: int = 5

    # --- Move Probabilities ---
    prob_bit_flip: float = 0.7  # Chance to flip a selection bit
    prob_route_swap: float = 0.3  # Chance to swap two nodes in the tour

    # --- Objective Weights ---
    overflow_penalty: float = 1000.0

    # --- Infrastructure ---
    seed: int = 42
    time_limit: float = 60.0

    @classmethod
    def from_config(cls, config: Any) -> "JointSAParams":
        """
        Build params from a config dataclass or dictionary.
        """
        if config is None:
            return cls()

        kwargs = {}
        # Handle both dicts and dataclasses (Hydra/OmegaConf)
        if hasattr(config, "__dict__"):
            source = config.__dict__
        elif hasattr(config, "items"):
            source = config
        else:
            return cls()

        for field in cls.__dataclass_fields__:
            if field in source and source[field] is not None:
                kwargs[field] = source[field]

        return cls(**kwargs)
