"""
HGS Policy Adapter.

Adapts the Hybrid Genetic Search (HGS) logic to the common policy interface.
Now agnostic to bin selection.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import HGSConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.hybrid_genetic_search import run_hgs

from .factory import PolicyRegistry


@PolicyRegistry.register("hgs")
class HGSPolicy(BaseRoutingPolicy):
    """
    Hybrid Genetic Search policy class.

    Visits pre-selected 'must_go' bins using evolutionary optimization.
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
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Run HGS solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        routes, profit, solver_cost = run_hgs(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            values,
            mandatory_nodes=mandatory_nodes,
        )
        return routes, profit, solver_cost
