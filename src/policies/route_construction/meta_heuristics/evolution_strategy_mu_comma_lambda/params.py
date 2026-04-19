from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MuCommaLambdaESParams:
    """
    Parameters for (μ,λ) Evolution Strategy with deterministic truncation selection.

    A (μ,λ)-ES maintains a parent population of size μ and generates λ offspring
    each iteration. This configuration ensures a memoryless stochastic process
    (Markov process), where the next parent population is derived strictly from
    the current offspring pool.

    Attributes:
        mu (int): The number of parent individuals maintained in the population (μ).
            A larger μ improves collective hillclimbing and reliability in
            multi-modal landscapes.
            Default follows standard big population settings.

        lambda_ (int): The number of offspring generated in each generation (λ).
            Standard (μ,λ) theory suggests that λ should be significantly
            larger than μ (e.g., λ ≈ 7μ) to ensure efficient step-size
            self-adaptation.

        n_removal (int): The mutation strength parameter. Defines the number
            of nodes removed during the destroy-repair mutation phase. This acts
            as the random perturbation added to components of the candidate
            solution.

        max_iterations (int): The generational loop limit. Serves as a
            primary termination criterion for the search process.

        local_search_iterations (int): The intensity of the local optimization
            applied to each offspring. This governs the fine-tuning of
            candidate solutions post-mutation.

        time_limit (float): Wall-clock duration in seconds. The algorithm
            will terminate early if the process time exceeds this threshold
            to ensure compliance with real-time operational constraints.
        seed (Optional[int]): Random seed for reproducibility.
        vrpp (bool): Whether the problem is VRP with Profits.
        profit_aware_operators (bool): Whether to use profit-aware insertion/removal.
    """

    mu: int = 15  # Parent population size (μ)
    lambda_: int = 100  # Offspring population size (λ)
    n_removal: int = 3  # Mutation strength
    max_iterations: int = 500
    local_search_iterations: int = 100
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    @classmethod
    def from_config(cls, config: Any) -> MuCommaLambdaESParams:
        """Build parameters from a configuration object."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in dataclasses.fields(cls)}})

        return cls(
            mu=getattr(config, "mu", 15),
            lambda_=getattr(config, "lambda_", 100),
            n_removal=getattr(config, "n_removal", 3),
            max_iterations=getattr(config, "max_iterations", 500),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
