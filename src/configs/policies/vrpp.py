"""
VRPP configuration.
"""

from dataclasses import dataclass


@dataclass
class VRPPConfig:
    engine: str = "gurobi"
    time_limit: float = 60.0
