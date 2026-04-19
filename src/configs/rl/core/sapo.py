"""SAPO specific configuration."""

from dataclasses import dataclass


@dataclass
class SAPOConfig:
    """SAPO specific configuration."""

    tau_pos: float = 0.1
    tau_neg: float = 1.0
