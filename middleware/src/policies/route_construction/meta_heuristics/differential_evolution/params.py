"""
Configuration parameters for the Memetic Differential Evolution (MDE) solver.

This module defines the structural constraints and hyper-parameters for MDE,
hybridizing the rigorous global search of Storn & Price (1997) with
Lamarckian/Baldwinian local exploitation for efficient VRPP optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DEParams:
    """
    Configuration parameters for Memetic Differential Evolution (MDE/rand/1/exp).

    Memetic Differential Evolution (MDE) hybridizes the exploratory power of
    the original DE/rand/1/exp (Storn & Price, 1997) with memetic local search.
    The core differential operators handle global search in the continuous Random
    Key space, while Lamarckian or Baldwinian strategies refine the discrete
    phenotypic solutions, an architectural addition to the original formulation.

    Attributes:
        pop_size (Optional[int]): Population size (NP). Number of candidate solution
            vectors. If None, scales dynamically as 10×D (where D is problem
            dimensionality) to satisfy the mutual exclusivity axiom.
            Recommended range (Storn & Price, 1997): 5×D to 10×D.

        mutation_factor (float): Differential weight (F). Scaling factor for the
            mutation vector. Controls the amplification of the differential variation.
            Range: [0, 2]. Standard: F ∈ [0.5, 1.0].
            - F = 0.5: Conservative exploration
            - F = 1.0: Aggressive exploration (classical DE)

        crossover_rate (float): Crossover probability (CR). Probability that a
            component is inherited from the mutant vector rather than the target.
            Range: [0, 1]. Standard: CR ∈ [0.8, 0.95].
            - CR = 0.0: No recombination (pure mutation)
            - CR = 1.0: Full replacement (no inheritance)

        n_removal (int): Mutation strength parameter for discrete operators.
            Number of nodes removed during destroy-repair mutation. This acts as
            the discrete analog to continuous mutation magnitude.

        max_iterations (int): Maximum number of generations (G_max).
            Primary termination criterion for the evolutionary loop.

        local_search_iterations (int): Intensity of local optimization applied
            to each trial vector. Governs the fine-tuning of candidate solutions
            post-mutation.

        evolution_strategy (str): Strategy for incorporating local search improvements.
            Options: "lamarckian", "baldwinian"
            - "lamarckian": Reverse-encode optimized routes back into continuous vector
              (genetic material is modified by local search)
            - "baldwinian": Keep original continuous vector, only use improved fitness
              (genetic material is unaffected by local search)
            Default: "lamarckian"

        time_limit (float): Wall-clock time limit in seconds. Algorithm terminates
            early if process time exceeds this threshold.

    Mathematical Foundation (Core Differential Evolution):
        MDE/rand/1/exp strategy:

        1. Mutation (Storn & Price, 1997):
           v_i = x_r1 + F × (x_r2 - x_r3)
           where r1, r2, r3 are distinct random indices ≠ i

        2. Crossover (Exponential):
           Inherit a consecutive sequence of parameters from v_i with
           probability CR until a random stop or full replacement.

        3. Memetic Integration (Extension):
           Local search (Lamarckian/Baldwinian) is applied after crossover
           to refine the discrete phenotypic solution before selection.

        4. Selection (Greedy):
           x_i(t+1) = u_i  if f(u_i) ≥ f(x_i)
                      x_i  otherwise

    Reference:
        Storn, R., & Price, K. (1997). "Differential Evolution – A Simple and
        Efficient Heuristic for Global Optimization over Continuous Spaces."
        Journal of Global Optimization, 11(4), 341-359.
    """

    pop_size: Optional[int] = 0  # Population size (NP), scales as 10×D if None or <= 0
    mutation_factor: float = 0.8  # Differential weight (F)
    crossover_rate: float = 0.9  # Crossover probability (CR)
    n_removal: int = 3  # Mutation strength for discrete operators
    max_iterations: int = 500
    local_search_iterations: int = 100
    evolution_strategy: str = "lamarckian"  # Evolution strategy: "lamarckian" or "baldwinian"
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    @classmethod
    def from_config(cls, config: Any) -> DEParams:
        """Build parameters from a configuration object."""
        return cls(
            pop_size=getattr(config, "pop_size", None),
            mutation_factor=getattr(config, "mutation_factor", 0.8),
            crossover_rate=getattr(config, "crossover_rate", 0.9),
            n_removal=getattr(config, "n_removal", 3),
            max_iterations=getattr(config, "max_iterations", 500),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            evolution_strategy=getattr(config, "evolution_strategy", "lamarckian"),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
