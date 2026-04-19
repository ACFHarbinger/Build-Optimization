"""
GA (Genetic Algorithm) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ga import GAConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import GAParams
from .solver import GASolver


@GlobalRegistry.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.POPULATION_BASED,
    PolicyTag.EVOLUTIONARY_ALGORITHM,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("ga")
class GAPolicy(BaseRoutingPolicy):
    """
    Genetic Algorithm (GA) Policy - Population-Based Evolutionary Routing.

    This policy employs a Genetic Algorithm to discover high-quality routing
    solutions through simulated evolution. It maintains a population of diverse
    routes (individuals) and iteratively applies biological operators:
    1.  **Selection**: Prioritizes "fitter" individuals (better profit/cost)
        for reproduction using Tournament or Roulette Wheel selection.
    2.  **Crossover**: Combines spatial features of two parent routes to
        produce offspring, preserving efficient sub-tours (e.g., OX, PMX).
    3.  **Mutation**: Introduces stochastic variations (e.g., swap, inverse,
        2-opt) to maintain population diversity and escape local optima.

    The GA is particularly effective at exploring large search spaces and
    parallelizing the optimization of complex multi-vehicle configurations.

    Registry key: ``"ga"``
    """

    def __init__(self, config: Optional[Union[GAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return GAConfig

    def _get_config_key(self) -> str:
        return "ga"

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
        Execute the Genetic Algorithm (GA) solver logic.

        GA is a population-based search heuristic inspired by the process of
        natural selection. It uses evolutionary operators such as selection,
        crossover (recombination), and mutation to evolve a population of
        candidate solutions. In this implementation:
        - Selection: Tournament selection is used to choose parents.
        - Crossover: Order-based crossover operators for route sequences.
        - Mutation: Random swaps or removals to maintain diversity.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                GA parameters (pop_size, crossover_rate, mutation_rate).
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
        params = GAParams(
            pop_size=int(values.get("pop_size", 30)),
            max_generations=int(values.get("max_generations", 100)),
            crossover_rate=float(values.get("crossover_rate", 0.8)),
            mutation_rate=float(values.get("mutation_rate", 0.1)),
            tournament_size=int(values.get("tournament_size", 3)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
            seed=values.get("seed", 42),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = GASolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()
