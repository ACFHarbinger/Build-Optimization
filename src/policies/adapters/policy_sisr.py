"""
SISR Policy Adapter.

Adapts the Slack Induction by String Removal (SISR) logic to the common policy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import SISRConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.slack_induction_by_string_removal import SISRParams, SISRSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("sisr")
class SISRPolicy(BaseRoutingPolicy):
    """
    Policy adapter for the SISR metaheuristic.
    """

    def __init__(self, config: Optional[Union[SISRConfig, Dict[str, Any]]] = None):
        """Initialize SISR policy with optional config.

        Args:
            config: SISRConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return SISRConfig

    def _get_config_key(self) -> str:
        return "sisr"

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
        Run SISR solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        cfg = self._config
        if cfg is not None:
            params = SISRParams(
                time_limit=cfg.time_limit,
                max_iterations=cfg.max_iterations,
                start_temp=cfg.start_temp,
                cooling_rate=cfg.cooling_rate,
                max_string_len=cfg.max_string_len,
                avg_string_len=cfg.avg_string_len,
                blink_rate=cfg.blink_rate,
                destroy_ratio=cfg.destroy_ratio,
            )
        else:
            params = SISRParams(
                time_limit=values.get("time_limit", 10.0),
                max_iterations=values.get("max_iterations", 1000),
                start_temp=values.get("start_temp", 100.0),
                cooling_rate=values.get("cooling_rate", 0.995),
                max_string_len=values.get("max_string_len", 10),
                avg_string_len=values.get("avg_string_len", 3.0),
                blink_rate=values.get("blink_rate", 0.01),
                destroy_ratio=values.get("destroy_ratio", 0.2),
            )

        solver = SISRSolver(sub_dist_matrix, sub_wastes, capacity, revenue, cost_unit, params)
        routes, profit, cost = solver.solve()
        return routes, profit, cost
