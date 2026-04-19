"""
SANS Policy Adapter (Simulated Annealing Neighborhood Search).

Uses Simulated Annealing for route optimization.
Supports two engines:
  - 'new': Improved SA with initial solution and iterative refinement
  - 'og': Original look-ahead algorithm for collection (LAC)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import SANSConfig
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.dispatcher import (
    execute_new,
    execute_og,
)

from .params import SANSParams


@RouteConstructorRegistry.register("sans")
@RouteConstructorRegistry.register("lac")  # Backward compatibility alias
class SANSPolicy(BaseRoutingPolicy):
    """
    Simulated Annealing Neighborhood Search policy class.

    Uses SA optimization with custom initialization and mandatory enforcement.
    Supports two engines via the 'engine' parameter:
      - 'new': Improved simulated annealing with initial solution computation
      - 'og': Original look-ahead collection (LAC) algorithm
    """

    def __init__(self, config: Optional[Union[SANSConfig, Dict[str, Any]]] = None):
        """Initialize SANS policy with optional config.

        Args:
            config: SANSConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return SANSConfig

    def _get_config_key(self) -> str:
        """Return config key for SANS."""
        return "sans"

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
        """Not used - SANS requires specialized execute()."""
        return [[]], 0.0, 0.0

    def execute(
        self, **kwargs: Any
    ) -> Tuple[List[int], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
        """
        Execute the SANS policy.

        Uses specialized data preparation for simulated annealing.
        """
        # Determine engine and parameters from typed config, raw config, or kwargs
        params = SANSParams.from_config(self._config or kwargs.get("config", {}).get("sans", {}))

        if params.engine == "og":
            return execute_og(self, params=params, **kwargs)
        else:
            return execute_new(self, params=params, **kwargs)
