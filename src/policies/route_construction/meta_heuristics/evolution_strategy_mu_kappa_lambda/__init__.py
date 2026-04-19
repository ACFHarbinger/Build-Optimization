"""
(μ,κ,λ) Evolution Strategy with age-based selection.

This module implements the classical (μ,κ,λ)-ES where selection occurs from
μ parents who have not exceeded an age of κ and λ offspring individuals.

Key components:
    - MuKappaLambdaESSolver: Main solver class
    - MuKappaLambdaESParams: Configuration parameters
    - Individual: Data structure for ES individuals with age tracking

Reference:
    Emmerich, M., Shir, O. M., & Wang, H. (2015). Evolution Strategies.
    In: Handbook of Natural Computing (pages 1-31).

Example:
    >>> import numpy as np
    >>> from logic.src.policies.evolution_strategy_mu_kappa_lambda import (
    ...     MuKappaLambdaESSolver,
    ...     MuKappaLambdaESParams
    ... )
    >>>
    >>> # Define sphere function
    >>> def sphere(x):
    ...     return np.sum(x**2)
    >>>
    >>> # Configure parameters
    >>> params = MuKappaLambdaESParams(
    ...     mu=15,
    ...     kappa=7,
    ...     lambda_=100,
    ...     rho=2,
    ...     max_iterations=100
    ... )
    >>>
    >>> # Solve
    >>> solver = MuKappaLambdaESSolver(
    ...     objective_function=sphere,
    ...     dimension=10,
    ...     params=params,
    ...     seed=42
    ... )
    >>> best_x, best_fitness = solver.solve()
    >>> print(f"Best fitness: {best_fitness:.6f}")
"""

from .params import MuKappaLambdaESParams
from .solver import Individual, MuKappaLambdaESSolver

__all__ = [
    "MuKappaLambdaESSolver",
    "MuKappaLambdaESParams",
    "Individual",
]
