"""
Hybrid Memetic Search (HMS) Policy Adapter.

Adapts the rigorous Hybrid Memetic Search (replaces HVPL).
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import HybridMemeticSearchConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params import ALNSParams
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import KSACOParams

from .params import HybridMemeticSearchParams
from .solver import HybridMemeticSearchSolver


@RouteConstructorRegistry.register("hms")
class HybridMemeticSearchPolicy(BaseRoutingPolicy):
    """
    Hybrid Memetic Search policy class.

    Multi-phase hybrid solver (ACO + GA + ALNS). Replaces HVPL.
    """

    def __init__(self, config: Optional[Union[HybridMemeticSearchConfig, Dict[str, Any]]] = None):
        """Initialize HMS policy with optional config.

        Args:
            config: HybridMemeticSearchConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HybridMemeticSearchConfig

    def _get_config_key(self) -> str:
        """Return config key."""
        return "hms"

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
        Execute the Hybrid Memetic Search (HMS) solver logic.

        HMS is a multi-phase hybrid metaheuristic that coordinates several
        optimization paradigms:
        - ACO Initialization: Generates a high-quality initial population.
        - Genetic Algorithm (GA) Core: Evolves the population through
          selection, crossover, and mutation.
        - ALNS Education: Refines offspring and elite solutions using adaptive
          local search operators.
        Replaces the earlier HVPL strategy with a more robust memetic framework
        governing the transitions between construction, evolution, and
        intensification phases.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                HMS parameters and nested ACO/ALNS configs.
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

        # Get ACO config (nested or flattened)
        aco_config = values.get("aco", {})
        if isinstance(aco_config, dict):
            aco_params = KSACOParams(
                n_ants=aco_config.get("n_ants", 20),
                k_sparse=aco_config.get("k_sparse", 10),
                alpha=aco_config.get("alpha", 1.0),
                beta=aco_config.get("beta", 2.0),
                rho=aco_config.get("rho", 0.1),
                q0=aco_config.get("q0", 0.9),
                tau_0=aco_config.get("tau_0", 1.0),
                tau_min=aco_config.get("tau_min", 0.001),
                tau_max=aco_config.get("tau_max", 10.0),
                max_iterations=aco_config.get("max_iterations", 1),
                time_limit=aco_config.get("time_limit", 60.0),
                local_search=aco_config.get("local_search", False),
                local_search_iterations=aco_config.get("local_search_iterations", 0),
                elitist_weight=aco_config.get("elitist_weight", 1.0),
                seed=seed,
                vrpp=vrpp,
                profit_aware_operators=profit_aware_operators,
            )
        else:
            # Fallback to flattened parameters for backward compatibility
            aco_params = KSACOParams(
                n_ants=values.get("aco_n_ants", 20),
                k_sparse=values.get("aco_k_sparse", 10),
                alpha=values.get("aco_alpha", 1.0),
                beta=values.get("aco_beta", 2.0),
                rho=values.get("aco_rho", 0.1),
                q0=values.get("aco_q0", 0.9),
                tau_0=values.get("aco_tau_0", 1.0),
                tau_min=values.get("aco_tau_min", 0.001),
                tau_max=values.get("aco_tau_max", 10.0),
                max_iterations=values.get("aco_iterations", 1),
                time_limit=values.get("aco_time_limit", 60.0),
                local_search=values.get("aco_local_search", False),
                local_search_iterations=values.get("aco_local_search_iterations", 0),
                elitist_weight=values.get("aco_elitist_weight", 1.0),
                seed=seed,
                vrpp=vrpp,
                profit_aware_operators=profit_aware_operators,
            )

        # Get ALNS config (nested or flattened)
        alns_config = values.get("alns", {})
        if isinstance(alns_config, dict):
            alns_params = ALNSParams(
                max_iterations=alns_config.get("max_iterations", 500),
                start_temp=alns_config.get("start_temp", 100.0),
                cooling_rate=alns_config.get("cooling_rate", 0.95),
                reaction_factor=alns_config.get("reaction_factor", 0.1),
                min_removal=alns_config.get("min_removal", 1),
                max_removal_pct=alns_config.get("max_removal_pct", 0.3),
                time_limit=alns_config.get("time_limit", 60.0),
                seed=seed,
                vrpp=vrpp,
                profit_aware_operators=profit_aware_operators,
            )
        else:
            # Fallback to flattened parameters for backward compatibility
            alns_params = ALNSParams(
                max_iterations=values.get("alns_iterations", 500),
                start_temp=values.get("alns_start_temp", 100.0),
                cooling_rate=values.get("alns_cooling_rate", 0.95),
                reaction_factor=values.get("alns_reaction_factor", 0.1),
                min_removal=values.get("alns_min_removal", 1),
                max_removal_pct=values.get("alns_max_removal_pct", 0.3),
                time_limit=values.get("alns_time_limit", 60.0),
                seed=seed,
                vrpp=vrpp,
                profit_aware_operators=profit_aware_operators,
            )

        params = HybridMemeticSearchParams(
            population_size=values.get("population_size", 30),
            max_generations=values.get("max_generations", 50),
            substitution_rate=values.get("substitution_rate", 0.2),
            crossover_rate=values.get("crossover_rate", 0.8),
            mutation_rate=values.get("mutation_rate", 0.1),
            elitism_count=values.get("elitism_count", 3),
            aco_init_iterations=values.get("aco_init_iterations", 50),
            time_limit=values.get("time_limit", 300.0),
            n_removal=values.get("n_removal", 3),
            aco_params=aco_params,
            alns_params=alns_params,
            seed=seed,
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
        )

        solver = HybridMemeticSearchSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        routes, profit, cost = solver.solve()
        return routes, profit, cost
