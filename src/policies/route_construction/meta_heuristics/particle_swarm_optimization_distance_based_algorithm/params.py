"""
Configuration parameters for Distance-Based Particle Swarm Optimization.

This replaces the metaphor-based "Firefly Algorithm" with standard PSO terminology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DistancePSOParams:
    """
    Parameters for Particle Swarm Optimization with velocity momentum.

    **TRUE PSO IMPLEMENTATION** with inertia-weighted velocity updates.
    Replaces the deceptive "Firefly Algorithm" which was merely PSO
    without velocity momentum terms.

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
        population_size: Number of particles in swarm.
        max_iterations: Maximum number of PSO iterations.

        # PSO Velocity Parameters (Kennedy & Eberhart 1995)
        inertia_weight_start: Initial inertia weight w(0) for exploration.
        inertia_weight_end: Final inertia weight w(T) for exploitation.
        cognitive_coef: c₁ - personal best acceleration constant.
        social_coef: c₂ - global best acceleration constant.

        # Continuous→Discrete Adaptation (Ai & Kachitvichyanukul 2009)
        n_removal: Number of nodes in velocity-based perturbation.
        velocity_to_mutation_rate: Scaling factor: velocity magnitude → mutation probability.

        # Local Search & Optimization
        local_search_iterations: Number of local search improvement steps.
        time_limit: Wall-clock time limit in seconds (0 = no limit).

        # Guided Insertion Weights
        alpha_profit: Weight for profit term in node insertion scoring.
        beta_will: Weight for willingness (fill level) term.
        gamma_cost: Weight for insertion cost penalty.

    Complexity:
        - Space: O(pop_size × n) for swarm storage + velocities + personal bests
        - Time per iteration: O(pop_size × n²) for velocity updates + insertions

    References:
        - Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization."
        - Ai, T. J., & Kachitvichyanukul, V. (2009). "A particle swarm optimization
          for the vehicle routing problem with simultaneous pickup and delivery."
    """

    # Swarm Configuration
    population_size: int = 20
    max_iterations: int = 500

    # PSO Velocity Momentum Parameters
    inertia_weight_start: float = 0.9  # w(0) - high for initial exploration
    inertia_weight_end: float = 0.4  # w(T) - low for final exploitation
    cognitive_coef: float = 2.0  # c₁ - personal best attraction (Ai & Kachitvichyanukul 2009)
    social_coef: float = 2.0  # c₂ - global best attraction

    # Discrete Adaptation Parameters
    n_removal: int = 3  # Nodes removed in velocity-based mutation
    velocity_to_mutation_rate: float = 0.1  # Velocity → mutation probability scaling

    # Optimization Parameters
    local_search_iterations: int = 100
    time_limit: float = 60.0

    # Guided Insertion Scoring Weights
    alpha_profit: float = 1.0
    beta_will: float = 0.5
    gamma_cost: float = 0.3

    # Profit-awareness
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = None

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

    @property
    def pop_size(self) -> int:
        """Alias for population_size."""
        return self.population_size

    @classmethod
    def from_config(cls, config: Any) -> "DistancePSOParams":
        """Create parameters from a configuration object."""
        return cls(
            population_size=getattr(config, "population_size", 20),
            max_iterations=getattr(config, "max_iterations", 500),
            inertia_weight_start=getattr(config, "inertia_weight_start", 0.9),
            inertia_weight_end=getattr(config, "inertia_weight_end", 0.4),
            cognitive_coef=getattr(config, "cognitive_coef", 2.0),
            social_coef=getattr(config, "social_coef", 2.0),
            n_removal=getattr(config, "n_removal", 3),
            velocity_to_mutation_rate=getattr(config, "velocity_to_mutation_rate", 0.1),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 60.0),
            alpha_profit=getattr(config, "alpha_profit", 1.0),
            beta_will=getattr(config, "beta_will", 0.5),
            gamma_cost=getattr(config, "gamma_cost", 0.3),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            seed=getattr(config, "seed", None),
        )
