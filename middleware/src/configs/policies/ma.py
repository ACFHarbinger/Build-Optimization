"""
Memetic Algorithm (MA) configuration for Hydra.

This module defines the type-safe schema for configuring the Memetic Algorithm
via Hydra YAML files. It uses dataclasses to ensure validated parameter injection.

Reference:
    Moscato, P., Cotta, C., & Mendes, A. (2004). "Memetic Algorithms".
    Reference: bibliography/Memetic_Algorithms.pdf
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class MAConfig:
    """
    Hydra configuration schema for the Memetic Algorithm policy.

    This class serves as the interface between YAML configuration files and the
    internal MAParams objects. It defines the structure and default values for
    configuring the hybrid evolutionary-local-search engine.

    Operational Architecture:
    - pop_size: Controls the breadth of search (population size).
    - Generations/Rates: Controls the depth and intensiveness of the search.
    - Constraints: Time limit and seed for optimization control.

    Attributes:
        pop_size: Number of solutions maintained in the population.
        max_generations: Total number of evolutionary cycles.
        crossover_rate: Probability of recombining parents.
        mutation_rate: Probability of random solution perturbation.
        local_search_rate: Probability of intensive hill-climbing on offspring.
        tournament_size: Pressure multiplier for competitive selection.
        n_removal: Perturbation magnitude for mutation.
        time_limit: Max execution time in seconds.
        seed: Randomness seed for reproducibility.
        vrpp: Flag indicating if the full profits-based problem is active.
        mandatory_selection: List of strategies for mandatory node selection.
        route_improvement: Optional solvers applied after the MA search.
    """

    # Core Evolutionary Hyper-parameters
    pop_size: int = 30
    max_generations: int = 100

    # Pipeline Operation Rates (Moscato Fig 3.2)
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    local_search_rate: float = 1.0  # Core characteristic of MAs

    # Structural parameters
    tournament_size: int = 3
    n_removal: int = 2

    # Resource Guarding
    time_limit: float = 60.0
    seed: Optional[int] = None

    # Simulator Compatibility Flags
    vrpp: bool = True
    profit_aware_operators: bool = False
    mandatory_selection: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
