"""
HGS (Hybrid Genetic Search) configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.acceptance_criteria import AcceptanceConfig, OnlyImprovingConfig
from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class HGSConfig:
    """Configuration for Hybrid Genetic Search (HGS) policy.
    Based on Vidal et al. (2022) - "Hybrid genetic search for the CVRP".

    Attributes:
        restart_timer: Maximum wall-clock seconds for optimization, before restarting the algorithm (0 = unlimited)
        time_limit: Maximum time in seconds for the solver (0.0 = unlimited).
        seed: Random seed for reproducibility.
        mu: Minimum population size per subpopulation.
        n_offspring: Generation size (number before survivor selection).
        nb_elite: Number of elite individuals to preserve.
        nb_close: Number of close individuals for diversity measurement.
        nb_granular: Granular search parameter for local search.
        target_feasible: Target proportion of feasible solutions.
        n_iterations_no_improvement: Max iterations without improvement.
        mutation_rate: Probability of applying local search to offspring.
        repair_probability: Probability of repairing infeasible offspring.
        crossover_rate: Probability of applying crossover.
        local_search_iterations: Number of local search iterations.
        use_cross_exchange: Whether to use cross exchange moves.
        use_lambda_interchange: Whether to use lambda interchange moves.
        lambda_max: Maximum lambda for lambda interchange moves.
        use_ejection_chains: Whether to use ejection chain moves.
        use_3opt: Whether to use 3-opt moves.
        max_vehicles: Maximum number of vehicles (0 for unlimited).
        initial_penalty_capacity: Initial penalty for capacity violations.
        penalty_increase: Multiplier for increasing penalty.
        penalty_decrease: Multiplier for decreasing penalty.
        engine: Solver engine to use ('custom', 'pyvrp').
        vrpp: Whether this is a VRPP problem.
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
    """

    # Core HGS parameters (Vidal 2022)
    time_limit: float = 0.0
    restart_timer: float = 0.0
    seed: Optional[int] = None
    mu: int = 25
    n_offspring: int = 40
    nb_elite: int = 4
    nb_close: int = 5
    nb_granular: int = 20
    target_feasible: float = 0.2
    n_iterations_no_improvement: int = 20000

    # Genetic operators
    mutation_rate: float = 1.0
    repair_probability: float = 0.5
    crossover_rate: float = 1.0

    # Local search
    local_search_iterations: int = 500
    max_vehicles: int = 0
    use_cross_exchange: bool = False
    use_lambda_interchange: bool = False
    lambda_max: int = 0
    use_ejection_chains: bool = False
    use_3opt: bool = False

    # Penalty management
    initial_penalty_capacity: float = 1.0
    penalty_increase: float = 1.2
    penalty_decrease: float = 0.85

    # Engine configuration
    engine: str = "custom"
    vrpp: bool = True
    profit_aware_operators: bool = False
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
    acceptance: AcceptanceConfig = AcceptanceConfig(method="only_improving", params=OnlyImprovingConfig())
