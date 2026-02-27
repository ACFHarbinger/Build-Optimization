"""
HMM-GD Policy Adapter.

Adapts the HMM + Great Deluge (HMM-GD) hyper-heuristic solver to the
agnostic BaseRoutingPolicy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.hmm_gd import HMMGDConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.hidden_markov_model_great_deluge.params import HMMGDParams
from logic.src.policies.hidden_markov_model_great_deluge.solver import HMMGDSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("hmm_gd")
class HMMGDPolicy(BaseRoutingPolicy):
    """
    HMM-GD policy class.

    Visits bins using the online-learning HMM + Great Deluge hyper-heuristic.
    The HMM learns which Low-Level Heuristic to apply based on observed search
    states (improving / stagnating / escaping).  The Great Deluge criterion
    provides acceptance control without temperature parameters.
    """

    def __init__(self, config: Optional[Union[HMMGDConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HMMGDConfig

    def _get_config_key(self) -> str:
        return "hmm_gd"

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
        params = HMMGDParams(
            max_iterations=int(values.get("max_iterations", 500)),
            flood_margin=float(values.get("flood_margin", 0.05)),
            rain_speed=float(values.get("rain_speed", 0.001)),
            learning_rate=float(values.get("learning_rate", 0.1)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 5)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = HMMGDSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        routes, profit, cost = solver.solve()
        return routes, profit, cost
