"""
(μ+λ) Evolution Strategy configuration.

Strictly follows the generational evolutionary algorithm terminology where:
- μ (mu): Parent population size.
- λ (lambda_): Offspring population size.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class MuPlusLambdaESConfig:
    """
    Configuration for (μ+λ) Evolution Strategy policy.

    The (μ+λ) scheme is an elitist generational strategy where the next parent
    population is selected from the union of the current parents and their
    offspring. This ensures monotonic improvement in solution quality.

    Attributes:
        mu (int): The number of individuals in the parent population (μ).
            These individuals are transferred to the next generation if they
            remain among the best μ solutions.

        lambda_ (int): The number of individuals in the offspring population (λ).
            Offspring are created via recombination and mutation variation
            operators.

        n_removal (int): The mutation strength parameter.
            Defines the number of nodes removed during the destroy-repair
            perturbation phase.

        max_iterations (int): The maximum number of evolution cycles
            (generations) to perform.

        local_search_iterations (int): Intensity of local optimization
            applied to each candidate offspring.

        vrpp (bool): Whether the problem is a VRP with Profits.

        time_limit (float): Maximum wall-clock duration for the search process
            in seconds.

        seed (Optional[int]): Random seed for deterministic reproducibility.

        mandatory_selection (Optional[List[MandatorySelectionConfig]]): Configuration
            for node selection strategies.

        route_improvement (Optional[List[RouteImprovingConfig]]): List of
            heuristics applied to refine solutions after optimization.
    """

    mu: int = 10
    lambda_: int = 5
    n_removal: int = 3
    max_iterations: int = 500
    local_search_iterations: int = 100
    time_limit: float = 60.0
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = None
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
