"""
HGS Policy Adapter.

Adapts the Hybrid Genetic Search (HGS) logic to the common policy interface.
Now agnostic to bin selection.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import HGSConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .dispatcher import run_hgs


@GlobalRegistry.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.POPULATION_BASED,
    PolicyTag.EVOLUTIONARY_ALGORITHM,
    PolicyTag.MEMETIC_SEARCH,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
    PolicyTag.PARALLELIZABLE,
)
@RouteConstructorRegistry.register("hgs")
class HGSPolicy(BaseRoutingPolicy):
    """
    Hybrid Genetic Search (HGS) Policy - State-of-the-Art Evolutionary Routing.

    This policy implements the Hybrid Genetic Search algorithm, specifically
    optimized for Capacitated Vehicle Routing Problems (CVRP) and variants. It
    is widely considered one of the most powerful heuristics for VRP due to its
    synergistic combination of population management and aggressive local search.

    Algorithm Logic:
    1.  **Exploration (Genetic)**: Maintains a population of solutions and
        generates offspring using advanced crossover operators (e.g., PIX, OX)
        that preserve spatial clusters and route segments.
    2.  **Intensification (Local Search)**: Every offspring undergoes a rigorous
        "education" phase using a highly optimized Local Search engine (SWAP*,
        Relocate, 2-Opt) to reach a local optimum.
    3.  **Diversity Management**: Employs a biased fitness function that rewards
        both quality (low cost) and diversity (uniqueness in the population),
        preventing premature convergence and maintaining a healthy search.

    HGS is the primary engine for solving large-scale, static routing instances
    within the framework.

    Registry key: ``"hgs"``
    """

    def __init__(self, config: Optional[Union[HGSConfig, Dict[str, Any]]] = None):
        """Initialize HGS policy with optional config.

        Args:
            config: HGSConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HGSConfig

    def _get_config_key(self) -> str:
        """Return config key for HGS."""
        return "hgs"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        x_coords: Optional[np.ndarray] = None,
        y_coords: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Execute the Hybrid Genetic Search (HGS) metaheuristic solver logic.

        HGS (specifically HGS-CVRP) is a state-of-the-art genetic algorithm for
        vehicle routing. It combines:
        - Advanced Genetic Operators: Pivot-based crossover (PIX) and large
          neighborhood movements.
        - Local Search: Intense refinement of offspring using localized local
          search (SWAP*, 2-OPT).
        - Diversity Management: Maintains a population of diverse and high-quality
          solutions, biased towards individuals that contribute to
          population diversity.
        This policy projects coordinates to a local plane to ensure correct polar
        sector angles during neighborhood pruning.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                HGS parameters (population_size, nb_elite, nb_granular).
            mandatory_nodes (List[int]): Local indices of bins that MUST be
                collected in this period.
            x_coords (Optional[np.ndarray]): Longitude coordinates for sector pruning.
            y_coords (Optional[np.ndarray]): Latitude coordinates for sector pruning.
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
        # Project lat/lng to a local equirectangular plane centred on the depot (index 0).
        # This is required for correct polar sector angles in SWAP* pruning.
        # Raw lat/lng cannot be used directly in atan2 — longitude degrees are shorter
        # than latitude degrees by a factor of cos(lat), causing ~22% distortion at 39°N.
        x_coords_proj: Optional[np.ndarray] = None
        y_coords_proj: Optional[np.ndarray] = None

        if x_coords is not None and y_coords is not None:
            depot_lat = float(y_coords[0])
            depot_lng = float(x_coords[0])
            cos_lat = math.cos(math.radians(depot_lat))
            x_coords_proj = (x_coords - depot_lng) * cos_lat
            y_coords_proj = y_coords - depot_lat

        routes, profit, solver_cost = run_hgs(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            values,
            mandatory_nodes=mandatory_nodes,
            x_coords=x_coords_proj,
            y_coords=y_coords_proj,
        )
        return routes, profit, solver_cost
