"""
Differential Evolution (DE/rand/1/bin) configuration.

**TRUE DE** replacing the Artificial Bee Colony (ABC) algorithm.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class DEConfig:
    """Configuration for Differential Evolution policy.

    **Replaces ABC** - Proper DE/rand/1/bin with rigorous mechanics.

    Core DE Components (Storn & Price 1997):
    - Differential mutation: v = x_r1 + F × (x_r2 - x_r3)
    - Binomial crossover with CR parameter
    - Greedy one-to-one selection
    - No metaphor (employed/onlooker/scout bees)
    - No trial counter or abandonment mechanism

    ABC is mathematically isomorphic to DE with fitness-proportionate selection
    instead of greedy selection, but with unnecessary "bee foraging" metaphor.

    DE Advantages over ABC:
    1. Greedy selection is faster than fitness-proportionate
    2. Explicit crossover operator (CR) controls exploration
    3. Simpler algorithm without metaphorical agent types
    4. Proven convergence properties
    5. 25+ years of theoretical foundation

    References:
        Storn, R., & Price, K. (1997). "Differential Evolution – A Simple and
        Efficient Heuristic for Global Optimization over Continuous Spaces."
        Journal of Global Optimization, 11(4), 341-359.

        Karaboga, D. (2005). "An idea based on honey bee swarm for numerical
        optimization." Technical Report TR06.
        [Note: ABC is DE with fitness-proportionate selection]
    """

    # Population Configuration
    pop_size: int = 50  # Population size (NP)

    # DE Parameters
    mutation_factor: float = 0.8  # Differential weight (F) ∈ [0, 2]
    crossover_rate: float = 0.9  # Crossover probability (CR) ∈ [0, 1]

    # Discrete Mutation Strength
    n_removal: int = 3  # Nodes removed during destroy-repair

    # Runtime Control
    max_iterations: int = 500
    local_search_iterations: int = 100
    evolution_strategy: str = "lamarckian"  # "lamarckian" or "baldwinian"
    time_limit: float = 60.0

    # Infrastructure
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
