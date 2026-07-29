"""
Configuration parameters for the Memetic Algorithm (MA) solver.

This module defines the parameters used by the MASolver, mapped to the framework
proposed in the seminal work:
    Moscato, P., Cotta, C., & Mendes, A. (2004). "Memetic Algorithms".
    Reference: bibliography/Memetic_Algorithms.pdf
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MAParams:
    """
    Hyper-parameters for the Memetic Algorithm (MA) solver.

    The paper defines a Memetic Algorithm as the synergistic combination of
    population-level evolutionary search (Genetic Algorithm) and individual-level
    intensive refinement (Local Search / Individual Search).

    Paper Concept Mapping:
    - pop_size: The constant size of the population denoted by |pop| in Fig. 3.1.
    - max_generations: The number of times the Generational Step is executed.
    - recombination: Controlled by crossover_rate within the reproduction pipeline (Fig. 3.2).
    - mutation: Controlled by mutation_rate and n_removal (p. 4).
    - local_improver: Intensive hill-climbing defined in Fig. 3.3.
    - replacement: Implements 'Plus' strategy for global elitism (p. 3).

    Attributes:
        pop_size: The number of active individuals in each generation (|pop|).
                  Determines the breadth of the global search.
        max_generations: The maximum number of generational iterations to perform.
        crossover_rate: The probability [0.0, 1.0] that two parents will
                         exchange information to create an offspring (Recombination).
        mutation_rate: The probability [0.0, 1.0] that a child will undergo
                       a random structural perturbation (Mutation).
        local_search_rate: The probability [0.0, 1.0] that the Local-Improver (Fig. 3.3)
                            will be applied to a newly generated offspring.
                            In many MAs, this is set to 1.0 for intensive search.
        tournament_size: The number of candidates selected for each fitness-based
                         competitive selection (Tournament Selection).
                         Higher values increase selection pressure.
        n_removal: The number of nodes removed and re-inserted during the
                   mutation operator. Represents the 'shake' magnitude.
        time_limit: Maximum wall-clock duration in seconds allowed for the search.
    """

    # Population and Iteration Settings
    pop_size: int = 30
    max_generations: int = 100

    # Probability Coefficients for Fig. 3.2 Pipeline
    crossover_rate: float = 0.8  # Probability of information exchange
    mutation_rate: float = 0.1  # Probability of random perturbation
    local_search_rate: float = 1.0  # Intensive search probability

    # Selection and Operator Scale
    tournament_size: int = 3  # Competitive selection pressure
    n_removal: int = 2  # Mutation aggressiveness

    # Resource Guard
    time_limit: float = 60.0  # Wall-clock timeout (seconds)
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = None

    @classmethod
    def from_config(cls, config: Any) -> "MAParams":
        """Create parameters from a configuration object."""
        return cls(
            pop_size=getattr(config, "pop_size", 30),
            max_generations=getattr(config, "max_generations", 100),
            crossover_rate=getattr(config, "crossover_rate", 0.8),
            mutation_rate=getattr(config, "mutation_rate", 0.1),
            local_search_rate=getattr(config, "local_search_rate", 1.0),
            tournament_size=getattr(config, "tournament_size", 3),
            n_removal=getattr(config, "n_removal", 2),
            time_limit=getattr(config, "time_limit", 60.0),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
