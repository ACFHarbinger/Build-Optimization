"""
SANS (Simulated Annealing Neighborhood Search) configuration.
"""

from typing import List, Optional

from .other.acceptance_criteria import AcceptanceConfig, BoltzmannAcceptanceConfig
from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


class SAConfig:
    """
    Configuration for the Simulated Annealing policy.

    Attributes:
        initial_temperature (float): Starting temperature T_0. Must be high enough
            to ensure an initial acceptance probability of approx 0.8 for worsening moves.
        cooling_rate (float): Geometric cooling coefficient alpha in (0, 1).
            T_{k+1} = alpha * T_k.
        min_temperature (float): Termination temperature threshold T_{min}.
        iterations_per_temp (int): Length of the Markov chain L_k at each temperature step.
            Allows the system to reach thermal equilibrium before cooling.
        nb_granular (int): Number of neighborhood search granular moves to try at each
            temperature step.
        time_limit (float): Absolute wall-clock time limit in seconds.
        seed (Optional[int]): Random seed for stochastic reproducibility.
        vrpp (bool): Flag to enable pool expansion for Vehicle Routing with Profits.
        profit_aware_operators (bool): Flag to bias ruin/recreate toward high-revenue nodes.
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
    """

    initial_temperature: float = 100.0
    cooling_rate: float = 0.95
    min_temperature: float = 0.01
    iterations_per_temp: int = 100
    nb_granular: int = 20
    time_limit: float = 60.0
    seed: Optional[int] = 42
    vrpp: bool = True
    profit_aware_operators: bool = False
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
    acceptance: AcceptanceConfig = AcceptanceConfig(
        method="boltzmann", params=BoltzmannAcceptanceConfig(initial_temp=100.0, alpha=0.95)
    )
