"""
Particle Swarm Optimization (PSO) with velocity momentum configuration.

**TRUE PSO** replacing the Sine Cosine Algorithm (SCA).
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class PSOConfig:
    """Configuration for PSO with velocity momentum policy.

    **Replaces SCA** - Proper PSO with all components intact.

    Core PSO Components (Kennedy & Eberhart 1995):
    - Velocity vectors with momentum (inertia term)
    - Personal best (pbest) tracking per particle (cognitive term)
    - Global best (gbest) for the swarm (social term)
    - Linearly decreasing inertia weight

    SCA is mathematically equivalent to PSO without velocity momentum,
    but with expensive trigonometric operations providing no benefit.

    References:
        Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization."
        Proceedings of ICNN'95 - International Conference on Neural Networks.

        Shi, Y., & Eberhart, R. (1998). "A modified particle swarm optimizer."
        IEEE International Conference on Evolutionary Computation.
    """

    # Swarm Configuration
    pop_size: int = 30

    # PSO Velocity Momentum Parameters
    inertia_weight_start: float = 0.9  # w(0) - high for exploration
    inertia_weight_end: float = 0.4  # w(T) - low for exploitation
    cognitive_coef: float = 2.0  # c₁ - personal best attraction
    social_coef: float = 2.0  # c₂ - global best attraction

    # Continuous Space Bounds
    position_min: float = -1.0  # Lower bound for positions
    position_max: float = 1.0  # Upper bound for positions
    velocity_max: float = 0.5  # Maximum velocity (clamping)

    # Runtime Control
    max_iterations: int = 500
    time_limit: float = 60.0

    # Infrastructure
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = None
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
