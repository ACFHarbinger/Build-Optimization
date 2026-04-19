"""
Configuration for the Genetic Programming Multi-Period Hyper-Heuristic (GP-MP-HH).
"""

from dataclasses import dataclass
from typing import Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class GP_MP_HH_Config:
    """
    Configuration for the Genetic Programming Multi-Period Hyper-Heuristic (GP-MP-HH) policy.

    GP-MP-HH evolves a sequence of Low-Level Heuristics (LLHs) using genetic
    programming to solve multi-period routing problems.

    Attributes:
        pop_size (int): Size of the GP population. Defaults to 10.
        gens (int): Number of generations. Defaults to 20.
        prog_len (int): Maximum length of a GP individual (program). Defaults to 5.
        seed (int): Random seed for reproducibility. Defaults to 42.
        vrpp (bool): Whether the problem is a VRP with Profits. Defaults to True.
        mandatory_selection (Optional[MandatorySelectionConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    pop_size: int = 10
    gens: int = 20
    prog_len: int = 5
    seed: int = 42
    vrpp: bool = True

    mandatory_selection: Optional[MandatorySelectionConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
