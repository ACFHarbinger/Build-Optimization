"""
MA (Memetic Algorithm) Policy Adapter.

This module provides the integration layer between the MASolver and the
broader simulator infrastructure. It adapts the evolutionary engine to the
agnostic BaseRoutingPolicy interface.

Reference:
    Moscato, P., Cotta, C., & Mendes, A. (2004). "Memetic Algorithms".
    Reference: bibliography/Memetic_Algorithms.pdf
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ma import MAConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import MAParams
from .solver import MASolver


@GlobalRegistry.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.POPULATION_BASED,
    PolicyTag.EVOLUTIONARY_ALGORITHM,
    PolicyTag.MEMETIC_SEARCH,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("ma")
class MAPolicy(BaseRoutingPolicy):
    """
    Memetic Algorithm (MA) Policy - Evolutionary Search with Individual Learning.

    This policy implements a Memetic Algorithm (Moscato, 1989), which extends the
    standard Genetic Algorithm by incorporating a local search (learning) phase
    for every individual offspring.

    Algorithm Components:
    1.  **Global Exploration**: Employs genetic operators (selection, crossover,
        mutation) to maintain a diverse population of routing solutions.
    2.  **Individual Education**: Each newly generated offspring undergoes a
        dedicated local search refinement (memetic learning) to reach a local
        optimum before being re-integrated into the population.
    3.  **Memetic Synergy**: This approach combines the broad exploratory power
        of evolutionary search with the precision of deterministic local
        optimization.

    Registry key: ``"ma"``
    """

    def __init__(self, config: Optional[Union[MAConfig, Dict[str, Any]]] = None):
        """
        Initialize the MAPolicy adapter.

        Args:
            config: An optional configuration object or dictionary containing
                    the engine hyper-parameters.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """
        Defines the Hydra-compliant configuration schema for this policy.

        Returns:
            MAConfig class.
        """
        return MAConfig

    def _get_config_key(self) -> str:
        """
        Returns the unique identifier for this policy in the registry.

        Returns:
            "ma"
        """
        return "ma"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Execute the Memetic Algorithm (MA) solver logic.

        MA is a population-based search heuristic that combines evolutionary
        global search with local search refinement. It is often described as
        a "Genetic Algorithm + Local Search". In this implementation:
        - Population: Maintains a set of candidate solutions.
        - Evolution: Standard genetic operators (selection, crossover, mutation)
          are applied to evolve the population.
        - Learning (Memetics): Each individual offspring undergoes a local search
          refinement (learning) before being added to the population, ensuring
          the search stays close to local optima.
        This policy acts as an adapter for the mathematical MASolver engine.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                MA parameters (pop_size, crossover_rate, mutation_rate, local_search_rate).
            mandatory_nodes (List[int]): Local indices of bins that MUST be
                collected in this period.
            **kwargs: Additional context, including:
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple[List[List[int]], float, float]: A 3-tuple containing:
                - routes: Optimized collection routes (list-of-lists, local indices).
                - profit: Total calculated net profit (Total Revenue - Total Cost).
                - cost: Total travel cost calculated by the solver.
        """
        # 1. Parameter Extraction & Mapping (See params.py for conceptual mapping)
        params = MAParams(
            pop_size=int(values.get("pop_size", 30)),
            max_generations=int(values.get("max_generations", 100)),
            crossover_rate=float(values.get("crossover_rate", 0.8)),
            mutation_rate=float(values.get("mutation_rate", 0.1)),
            local_search_rate=float(values.get("local_search_rate", 1.0)),
            tournament_size=int(values.get("tournament_size", 3)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        # 2. Solver Initialization
        solver = MASolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        # 3. Evolutionary Optimization (Moscato FIG 3.1)
        # Returns the best individual and its fitness (profit).
        best_routes, best_reward = solver.solve()

        # 4. Result Formatting & Supplemental Metrics
        # The simulator Expects (Routes, NetProfit, AbsoluteDistanceCost).
        total_cost = solver._cost(best_routes) * cost_unit

        return best_routes, best_reward, total_cost
