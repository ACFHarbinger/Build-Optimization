"""
Reinforcement Learning Great Deluge Hyper-Heuristic (RL-GD-HH) for VRPP.

This module implements the hyper-heuristic framework described in:
    Ozcan, E., Misir, M., Ochoa, G., & Burke, E. K. (2010).
    "A Reinforcement Learning – Great-Deluge Hyper-heuristic for Examination Timetabling".
    Reference: bibliography/Reinforcement_Learning_Great_Deluge_Hyper-Heuristic.pdf

The solver integrates:
1. Reinforcement Learning (RL) for online selection of low-level heuristics (LLHs).
2. Great Deluge (GD) algorithm for deterministic, time-based threshold acceptance.
"""

from .params import RLGDHHParams
from .policy_rl_gd_hh import RLGDHHPolicy
from .solver import RLGDHHSolver

__all__ = [
    "RLGDHHSolver",
    "RLGDHHParams",
    "RLGDHHPolicy",
]
