"""
Configuration parameters for Particle Swarm Optimization (PSO).

**TRUE PSO** with inertia-weighted velocity updates (Kennedy & Eberhart 1995).
This serves as a rigorous replacement for the Sine Cosine Algorithm (SCA),
which is mathematically equivalent to PSO without velocity momentum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PSOParams:
    """
    Parameters for standard Particle Swarm Optimization with velocity momentum.

    **Replaces SCA** - The Sine Cosine Algorithm is mathematically identical to
    PSO without the velocity term, personal best tracking, and with expensive
    trigonometric functions replacing simple random weights.

    Core PSO Velocity Update Equation (Kennedy & Eberhart 1995):
        v(t+1) = w*v(t) + c₁*r₁*(pbest - x(t)) + c₂*r₂*(gbest - x(t))
        x(t+1) = x(t) + v(t+1)

    Where:
        - w: inertia weight (balances exploration vs exploitation)
        - v(t): current velocity vector
        - c₁: cognitive acceleration coefficient (personal best attraction)
        - c₂: social acceleration coefficient (global best attraction)
        - r₁, r₂: random numbers ∈ [0,1]
        - pbest: particle's personal best position
        - gbest: swarm's global best position
        - x(t): current particle position

    Attributes:
        pop_size: Number of particles in swarm.
        max_iterations: Maximum number of PSO iterations.

        # PSO Velocity Parameters (Kennedy & Eberhart 1995)
        inertia_weight_start: Initial inertia weight w(0) for exploration.
        inertia_weight_end: Final inertia weight w(T) for exploitation.
        cognitive_coef: c₁ - personal best acceleration constant.
        social_coef: c₂ - global best acceleration constant.

        # Continuous Space Parameters
        position_min: Minimum value for position vector components.
        position_max: Maximum value for position vector components.
        velocity_max: Maximum velocity magnitude (velocity clamping).

        # Runtime Control
        time_limit: Wall-clock time limit in seconds (0 = no limit).

    Complexity:
        - Space: O(pop_size × n) for swarm + velocities + personal bests
        - Time per iteration: O(pop_size × n) for velocity updates

    Why PSO > SCA:
        1. **Velocity Momentum**: PSO maintains inertia from previous iterations
        2. **Personal Best**: Each particle learns from its own experience
        3. **No Expensive Trigonometry**: sin(r) ≈ random(-1,1) but slower
        4. **Proven Theory**: PSO has 30+ years of theoretical foundation
        5. **Better Convergence**: Velocity helps escape local optima

    References:
        Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization."
        Proceedings of ICNN'95 - International Conference on Neural Networks.

        Shi, Y., & Eberhart, R. (1998). "A modified particle swarm optimizer."
        IEEE International Conference on Evolutionary Computation.
    """

    # Swarm Configuration
    pop_size: int = 30

    # PSO Velocity Momentum Parameters
    inertia_weight_start: float = 0.9  # w(0) - high for initial exploration
    inertia_weight_end: float = 0.4  # w(T) - low for final exploitation
    cognitive_coef: float = 2.0  # c₁ - personal best attraction
    social_coef: float = 2.0  # c₂ - global best attraction

    # Continuous Space Bounds
    position_min: float = -1.0  # Lower bound for position vector
    position_max: float = 1.0  # Upper bound for position vector
    velocity_max: float = 0.5  # Maximum velocity (prevents divergence)

    # Runtime Control
    max_iterations: int = 500
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    def get_inertia_weight(self, iteration: int) -> float:
        """
        Compute linearly decreasing inertia weight.

        High inertia weight early → exploration (particles move freely).
        Low inertia weight late → exploitation (particles converge to best).

        Args:
            iteration: Current iteration number (0-indexed).

        Returns:
            Inertia weight w(t) ∈ [w_end, w_start].

        Formula (Shi & Eberhart 1998):
            w(t) = w_max - (w_max - w_min) × (t / T_max)
        """
        if self.max_iterations <= 1:
            return self.inertia_weight_end

        decay_factor = iteration / (self.max_iterations - 1)
        return self.inertia_weight_start - (self.inertia_weight_start - self.inertia_weight_end) * decay_factor

    @property
    def c1(self) -> float:
        """Alias for cognitive_coef (standard PSO notation)."""
        return self.cognitive_coef

    @property
    def c2(self) -> float:
        """Alias for social_coef (standard PSO notation)."""
        return self.social_coef

    @property
    def w_start(self) -> float:
        """Alias for inertia_weight_start."""
        return self.inertia_weight_start

    @property
    def w_end(self) -> float:
        """Alias for inertia_weight_end."""
        return self.inertia_weight_end

    @classmethod
    def from_config(cls, config: Any) -> "PSOParams":
        """Create parameters from a configuration object."""
        return cls(
            pop_size=getattr(config, "pop_size", 30),
            inertia_weight_start=getattr(config, "inertia_weight_start", 0.9),
            inertia_weight_end=getattr(config, "inertia_weight_end", 0.4),
            cognitive_coef=getattr(config, "cognitive_coef", 2.0),
            social_coef=getattr(config, "social_coef", 2.0),
            position_min=getattr(config, "position_min", -1.0),
            position_max=getattr(config, "position_max", 1.0),
            velocity_max=getattr(config, "velocity_max", 0.5),
            max_iterations=getattr(config, "max_iterations", 500),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
