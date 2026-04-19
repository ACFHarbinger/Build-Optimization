"""
HGS-ALNS Hybrid Policy Adapter.

Adapts the Hybrid HGS-ALNS solver to the common simulator policy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import HGSALNSConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params import ALNSParams
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.params import HGSParams
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.hgs_alns import (
    HGSALNSSolver,
)
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params import (
    HGSALNSParams,
)


@RouteConstructorRegistry.register("hgs_alns")
class HGSALNSPolicy(BaseRoutingPolicy):
    """
    Hybrid Genetic Search with Adaptive Large Neighborhood Search (HGS-ALNS) Policy.

    This policy implements a high-performance hybrid metaheuristic that combines
    the population-based evolutionary search of HGS with the multi-neighborhood
    intensification of ALNS.

    Architecture:
    1.  **Genetic Core**: Maintains a population of high-quality solutions and
        uses advanced crossover (e.g., SISR-based or Ordered crossover) to
        generate new offspring.
    2.  **ALNS Education**: Instead of standard local search, each newly generated
        offspring undergoes an "education" phase powered by an Adaptive Large
        Neighborhood Search engine.
    3.  **Diversity Management**: Continuously monitors population diversity
        and removes individuals that are too similar or have poor fitness, ensuring
        a robust exploration of the search space.

    Registry key: ``"hgs_alns"``
    """

    def __init__(self, config: Optional[Union[HGSALNSConfig, Dict[str, Any]]] = None):
        """Initialize HGS-ALNS policy with optional config.

        Args:
            config: HGSALNSConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HGSALNSConfig

    def _get_config_key(self) -> str:
        """Return config key for HGS-ALNS hybrid."""
        return "hgs_alns"

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
        Execute the Hybrid Genetic Search - Adaptive Large Neighborhood Search
        (HGS-ALNS) solver logic.

        HGS-ALNS is a hyper-hybrid metaheuristic that integrates ALNS into the
        HGS framework. Specifically, ALNS is used as an "education" (local
        search) operator within the HGS evolutionary cycle. This leverages the
        broad exploratory power of HGS's population management and crossover
        while utilizing the flexible, multi-neighborhood refinement capabilities
        of ALNS to intensify the search around promising solution areas.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                nested HGS and ALNS parameters.
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
        seed = values.get("seed", 42)
        vrpp = values.get("vrpp", True)
        profit_aware_operators = values.get("profit_aware_operators", False)

        # Build Params from nested config
        alns_params = ALNSParams(
            max_iterations=values.get("alns_iterations", 500),
            start_temp=values.get("alns_start_temp", 100.0),
            cooling_rate=values.get("alns_cooling_rate", 0.95),
            reaction_factor=values.get("alns_reaction_factor", 0.1),
            min_removal=values.get("alns_min_removal", 1),
            max_removal_pct=values.get("alns_max_removal_pct", 0.2),
            time_limit=values.get("alns_time_limit", 60.0),
            engine=values.get("alns_engine", "custom"),
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
            seed=seed,
        )

        hgs_params = HGSParams(
            time_limit=values.get("hgs_time_limit", 60.0),
            mu=values.get("hgs_population_size", 50),
            nb_elite=values.get("hgs_elite_size", 5),
            mutation_rate=values.get("hgs_mutation_rate", 0.2),
            crossover_rate=values.get("hgs_crossover_rate", 0.7),
            n_offspring=values.get("hgs_n_generations", 100),  # Mapping generations to offspring for this adapter
            n_iterations_no_improvement=values.get("hgs_no_improvement_threshold", 20),
            nb_granular=values.get("hgs_neighbor_list_size", 10),
            local_search_iterations=values.get("hgs_local_search_iterations", 500),
            max_vehicles=values.get("hgs_max_vehicles", 0),
            engine=values.get("hgs_engine", "custom"),
            seed=seed,
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
            restart_timer=values.get("hgs_restart_timer", 0.0),
        )

        # Create HGSALNSParams
        params = HGSALNSParams(
            time_limit=values.get("time_limit", 60.0),
            hgs_params=hgs_params,
            alns_params=alns_params,
            seed=seed,
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
        )

        solver = HGSALNSSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        routes, profit, solver_cost = solver.solve()
        return routes, profit, solver_cost
