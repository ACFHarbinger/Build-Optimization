"""
Configuration for the Population Hyper-Heuristic (PHH).
"""

from dataclasses import dataclass
from typing import Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class PHHConfig:
    """
    Configuration for the Population Hyper-Heuristic (PHH) policy.

    PHH maintains a population of multi-period plans and evolves them
    using Low-Level Heuristics (LLHs) and genetic operators.

    Attributes:
        pop_size (int): Size of the population. Defaults to 10.
        gens (int): Number of generations. Defaults to 20.
        seed (int): Random seed for reproducibility. Defaults to 42.
        vrpp (bool): Whether the problem is a VRP with Profits. Defaults to True.
        mandatory_selection (Optional[MandatorySelectionConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    pop_size: int = 10
    gens: int = 20
    seed: int = 42
    vrpp: bool = True

    mandatory_selection: Optional[MandatorySelectionConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
