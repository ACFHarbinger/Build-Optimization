"""
Configuration parameters for Hybrid Genetic Search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion

if TYPE_CHECKING:
    from logic.src.configs.policies import HGSConfig


@dataclass
class HGSParams:
    """
    Configuration parameters for Hybrid Genetic Search.
    Based on Vidal et al. (2022) - "Hybrid genetic search for the CVRP".

    Attributes:
        restart_timer: Maximum wall-clock seconds for optimization, before restarting the algorithm (0 = unlimited)
        time_limit: Maximum search time in seconds (0 = unlimited).
        mu: Minimum population size (for each subpopulation).
        lambda_param: Generation size - number of individuals before survivor selection.
        nb_elite: Number of elite individuals to preserve.
        nb_close: Number of close individuals for diversity measurement.
        nb_granular: Granular search parameter for local search moves.
        target_feasible: Target proportion of feasible solutions (e.g., 0.2 = 20%).
        n_iterations_no_improvement: Max iterations without improvement before stopping.
        restart_timer: Iterations without improvement before restarting (used only when time_limit > 0). Set to 0 to disable restarts.
        mutation_rate: Probability of applying local search (education) to offspring.
        repair_probability: Probability of repairing infeasible offspring (default 0.5).
        crossover_rate: Probability of applying crossover.
        local_search_iterations: Number of iterations for local search.
        use_cross_exchange: Whether to use cross exchange moves.
        use_lambda_interchange: Whether to use lambda interchange moves.
        lambda_max: Maximum lambda for lambda interchange moves.
        use_ejection_chains: Whether to use ejection chain moves.
        use_3opt: Whether to use 3-opt moves.
        max_vehicles: Maximum number of vehicles allowed (0 = unlimited).
        initial_penalty_capacity: Initial penalty coefficient for capacity violations.
        penalty_increase: Multiplier for increasing penalty (when too many feasible).
        penalty_decrease: Multiplier for decreasing penalty (when too many infeasible).
        engine: Engine to use for the solver.
    """

    # Core HGS parameters (Vidal 2022)
    restart_timer: float = 0.0
    time_limit: float = 0.0  # 0 = no time limit
    mu: int = 25  # Minimum population size per subpopulation
    n_offspring: int = 40  # Generation size (number of individuals before survivor selection)
    nb_elite: int = 4  # Number of elite individuals
    nb_close: int = 5  # Number of close individuals for diversity
    nb_granular: int = 20  # Granular search parameter
    target_feasible: float = 0.2  # Target 20% feasible solutions
    n_iterations_no_improvement: int = 20000  # Stopping criterion

    # Genetic operators
    mutation_rate: float = 1.0  # Always educate offspring with local search
    repair_probability: float = 0.5  # 50% chance to repair infeasible offspring
    crossover_rate: float = 1.0  # Always apply crossover

    # Local search
    local_search_iterations: int = 500
    max_vehicles: int = 0

    # Penalty management
    initial_penalty_capacity: float = 1.0
    penalty_increase: float = 1.2
    penalty_decrease: float = 0.85
    use_3opt: bool = False
    use_cross_exchange: bool = False
    use_lambda_interchange: bool = False
    lambda_max: int = 0
    use_ejection_chains: bool = False

    # Infrastructure
    seed: Optional[int] = None
    vrpp: bool = True
    engine: str = "custom"
    profit_aware_operators: bool = False
    acceptance_criterion: Optional[IAcceptanceCriterion] = None

    @classmethod
    def from_config(cls, config: HGSConfig) -> HGSParams:
        """Create HGSParams from a HGSConfig dataclass.

        Args:
            config: HGSConfig dataclass with solver parameters.

        Returns:
            HGSParams instance with values from config.
        """
        # Map config parameters to HGSParams, using defaults for new parameters
        params = cls(
            time_limit=getattr(config, "time_limit", 0.0),
            mu=getattr(config, "mu", 25),
            n_offspring=getattr(config, "n_offspring", getattr(config, "lambda_param", 40)),
            nb_elite=getattr(config, "nb_elite", 4),
            nb_close=getattr(config, "nb_close", 5),
            nb_granular=getattr(config, "nb_granular", 20),
            target_feasible=getattr(config, "target_feasible", 0.2),
            n_iterations_no_improvement=getattr(config, "n_iterations_no_improvement", 20000),
            mutation_rate=getattr(config, "mutation_rate", 1.0),
            repair_probability=getattr(config, "repair_probability", 0.5),
            crossover_rate=getattr(config, "crossover_rate", 1.0),
            local_search_iterations=getattr(config, "local_search_iterations", 500),
            max_vehicles=getattr(config, "max_vehicles", 0),
            initial_penalty_capacity=getattr(config, "initial_penalty_capacity", 1.0),
            penalty_increase=getattr(config, "penalty_increase", 1.2),
            penalty_decrease=getattr(config, "penalty_decrease", 0.85),
            use_3opt=getattr(config, "use_3opt", False),
            use_cross_exchange=getattr(config, "use_cross_exchange", False),
            use_lambda_interchange=getattr(config, "use_lambda_interchange", False),
            lambda_max=getattr(config, "lambda_max", 0),
            use_ejection_chains=getattr(config, "use_ejection_chains", False),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            seed=getattr(config, "seed", 42),
            engine=getattr(config, "engine", "custom"),
            restart_timer=getattr(config, "restart_timer", 0.0),
        )

        # Handle Acceptance Criterion Injection
        from logic.src.policies.route_construction.acceptance_criteria.base.factory import AcceptanceCriterionFactory

        acceptance_cfg = getattr(config, "acceptance", None)
        if acceptance_cfg:
            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name=acceptance_cfg.method,
                config=acceptance_cfg.params,
            )
        else:
            # Default to only_improving for standard HGS
            params.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")

        return params

    @property
    def lambda_param(self) -> int:
        """Alias for n_offspring (Vidal 2022 terminology)."""
        return self.n_offspring

    @lambda_param.setter
    def lambda_param(self, value: int):
        self.n_offspring = value

    @property
    def population_size(self) -> int:
        """Alias for mu (common evolutionary terminology)."""
        return self.mu

    @population_size.setter
    def population_size(self, value: int):
        self.mu = value

    @property
    def elite_size(self) -> int:
        """Alias for nb_elite."""
        return self.nb_elite

    @elite_size.setter
    def elite_size(self, value: int):
        self.nb_elite = value

    @property
    def no_improvement_threshold(self) -> int:
        """Alias for n_iterations_no_improvement."""
        return self.n_iterations_no_improvement

    @no_improvement_threshold.setter
    def no_improvement_threshold(self, value: int):
        self.n_iterations_no_improvement = value

    @property
    def neighbor_list_size(self) -> int:
        """Alias for nb_granular."""
        return self.nb_granular

    @neighbor_list_size.setter
    def neighbor_list_size(self, value: int):
        self.nb_granular = value
