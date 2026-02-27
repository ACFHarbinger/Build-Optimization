"""
OBA (Old Bachelor Acceptance) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.oba import OBAConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.old_bachelor_acceptance.params import OBAParams
from logic.src.policies.old_bachelor_acceptance.solver import OBASolver

from .factory import PolicyRegistry


@PolicyRegistry.register("oba")
class OBAPolicy(BaseRoutingPolicy):
    """Old Bachelor Acceptance policy class."""

    def __init__(self, config: Optional[Union[OBAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return OBAConfig

    def _get_config_key(self) -> str:
        return "oba"

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
        params = OBAParams(
            dilation=float(values.get("dilation", 5.0)),
            contraction=float(values.get("contraction", 2.0)),
            max_iterations=int(values.get("max_iterations", 500)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 5)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = OBASolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()
