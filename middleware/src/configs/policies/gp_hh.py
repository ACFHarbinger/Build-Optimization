"""
GPHH (Genetic Programming Hyper-Heuristic) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class GPHHConfig:
    """
    Configuration for the GPHH Constructive Heuristic Generator policy.

    Attributes:
        gp_pop_size: GP meta-population size (number of scoring trees).
        max_gp_generations: GP evolution generations.
        tree_depth: Maximum depth of GP expression trees.
        tournament_size: Tournament size for GP parent selection.
        time_limit: Wall-clock time limit in seconds.
        parsimony_coefficient: Weight for tree-size penalty (0.0 = none).
        n_training_instances: Sub-instances for fitness averaging.
        training_sample_ratio: Fraction of nodes sampled per training instance.
        vrpp: If True, solver operates in full VRPP mode (optional nodes).
        mandatory_selection: Mandatory selection strategy config list.
        route_improvement: Route improvement operation config list.
    """

    gp_pop_size: int = 20
    max_gp_generations: int = 30
    tree_depth: int = 3
    tournament_size: int = 3
    time_limit: float = 60.0
    parsimony_coefficient: float = 0.0
    n_training_instances: int = 3
    training_sample_ratio: float = 0.5
    seed: Optional[int] = None
    vrpp: bool = True
    mandatory_selection: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
