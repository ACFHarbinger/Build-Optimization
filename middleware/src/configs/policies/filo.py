"""
FILO (Fast Iterative Localized Optimization) configuration dataclasses.
"""

from dataclasses import dataclass

from .abc import ABCConfig


@dataclass
class FILOConfig(ABCConfig):
    """
    Configuration parameters for the Fast Iterative Localized Optimization (FILO) solver.

    Attributes:
        time_limit: Maximum allowed runtime in seconds.
        max_iterations: Maximum number of core optimization iterations.
        initial_temperature_factor: Divisor for initial SA temperature.
        final_temperature_factor: Divisor for final SA temperature.
        shaking_lb_factor: Lower bound intensification factor for shaking.
        shaking_ub_factor: Upper bound intensification factor for shaking.
        shaking_lb_intensity: Lower intensity bound for adaptive omega.
        shaking_ub_intensity: Upper intensity bound for adaptive omega.
        omega_alpha: Probability of jumping to a neighbor route during ruin.
        omega_base_multiplier: Base shaking intensity multiplier.
        delta_gamma: Coefficient for dynamic gamma threshold calculation.
        gamma_base: Base probability for neighborhood evaluation.
        gamma_lambda: Expansion factor for stagnant nodes.
        svc_size: Maximum size of the Selective Vertex Cache.
        n_cw: Number of nearest neighbors for Clarke-Wright savings.
        local_search_iterations: Maximum local search iterations per call.
        seed: Random seed for reproducibility.
        vrpp: Whether this is a VRPP problem.
        profit_aware_operators: Whether to use profit-aware destroy/repair operators.
    """

    time_limit: float = 60.0
    max_iterations: int = 50000
    initial_temperature_factor: float = 10.0
    final_temperature_factor: float = 100.0
    shaking_lb_factor: float = 0.5
    shaking_ub_factor: float = 2.0
    shaking_lb_intensity: float = 0.5
    shaking_ub_intensity: float = 1.5
    omega_alpha: float = 0.5
    omega_base_multiplier: float = 1.0
    delta_gamma: float = 0.25
    gamma_base: float = 1.0
    gamma_lambda: float = 2.0
    svc_size: int = 50
    n_cw: int = 100
    local_search_iterations: int = 500
    seed: int = 42
    vrpp: bool = True
    profit_aware_operators: bool = False
