from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MuPlusLambdaESParams:
    r"""
    Parameters for the (μ+λ) Evolution Strategy.

    A (μ+λ)-ES maintains a population of μ parent solutions. In each iteration,
    it generates λ offspring through recombination and mutation.
    The selection operator then selects the absolute best μ individuals from the
    combined pool of μ parents and λ offspring to form the next generation.

    Attributes:
        mu (int): The number of parent individuals maintained in the population ($\mu$).
            This acts as an archive of the best-found solutions, guaranteeing
            strong elitism and monotonic improvement.

        lambda_ (int): The number of offspring generated per generation ($\lambda$).
            This defines the exploration capacity per cycle.

        n_removal (int): The mutation strength parameter. Defines the number
            of nodes removed during the destroy-repair mutation phase.

        max_iterations (int): The generational loop limit. Serves as a
            primary termination criterion for the search process.

        local_search_iterations (int): The intensity of the local optimization
            applied to each offspring.

        time_limit (float): Wall-clock duration in seconds. The algorithm
            will terminate early if the process time exceeds this threshold.
        seed (Optional[int]): Random seed for reproducibility.
        vrpp (bool): Whether the problem is VRP with Profits.
        profit_aware_operators (bool): Whether to use profit-aware insertion/removal.
    """

    mu: int = 10  # Parent population size (μ)
    lambda_: int = 5  # Offspring population size (λ)
    n_removal: int = 3  # Mutation strength
    max_iterations: int = 500
    local_search_iterations: int = 100
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    @classmethod
    def from_config(cls, config: Any) -> MuPlusLambdaESParams:
        """Build parameters from a configuration object."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in dataclasses.fields(cls)}})

        return cls(
            mu=getattr(config, "mu", 10),
            lambda_=getattr(config, "lambda_", 5),
            n_removal=getattr(config, "n_removal", 3),
            max_iterations=getattr(config, "max_iterations", 500),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
