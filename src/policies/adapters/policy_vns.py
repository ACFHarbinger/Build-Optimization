"""
VNS (Variable Neighborhood Search) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.vns import VNSConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.variable_neighborhood_search.params import VNSParams
from policies.variable_neighborhood_search.solver import VNSSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("vns")
class VNSPolicy(BaseBuildPolicy):
    """Variable Neighborhood Search policy adapter."""

    def __init__(self, config: Optional[Union[VNSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return VNSConfig

    def _get_config_key(self) -> str:
        return "vns"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = VNSParams(
            k_max=int(values.get("k_max", 5)),
            max_iterations=int(values.get("max_iterations", 200)),
            local_search_iterations=int(values.get("local_search_iterations", 20)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = VNSSolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
