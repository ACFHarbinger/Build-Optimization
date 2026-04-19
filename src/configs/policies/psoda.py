"""
Particle Swarm Optimization with Velocity Momentum configuration.

**TRUE PSO** implementation with inertia-weighted velocity updates.
Replaces the deceptive "Firefly Algorithm" which lacked velocity momentum.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class DistancePSOConfig:
    """Configuration for PSO with velocity momentum policy.

    **TRUE PARTICLE SWARM OPTIMIZATION** (Kennedy & Eberhart 1995).

    Core PSO Components:
    - Velocity vectors with momentum (inertia term)
    - Personal best (pbest) tracking per particle (cognitive term)
    - Global best (gbest) for the swarm (social term)
    - Linearly decreasing inertia weight

    References:
        Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization."
        Ai, T. J., & Kachitvichyanukul, V. (2009). "A particle swarm optimization
        for the vehicle routing problem with simultaneous pickup and delivery."
    """

    # Swarm Configuration
    population_size: int = 20

    # PSO Velocity Momentum Parameters
    inertia_weight_start: float = 0.9  # w(0) - high for exploration
    inertia_weight_end: float = 0.4  # w(T) - low for exploitation
    cognitive_coef: float = 2.0  # c₁ - personal best attraction
    social_coef: float = 2.0  # c₂ - global best attraction

    # Discrete Adaptation Parameters
    n_removal: int = 3  # Nodes in velocity perturbation
    velocity_to_mutation_rate: float = 0.1  # Velocity → mutation scaling

    # Optimization Parameters
    max_iterations: int = 500
    local_search_iterations: int = 100
    time_limit: float = 60.0

    # Guided Insertion Scoring Weights
    alpha_profit: float = 1.0
    beta_will: float = 0.5
    gamma_cost: float = 0.3

    # Infrastructure
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = None
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
