"""
HS (Harmony Search) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.hs import HSConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.harmony_search.params import HSParams
from policies.harmony_search.solver import HSSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("hs")
class HSPolicy(BaseBuildPolicy):
    """Harmony Search policy adapter."""

    def __init__(self, config: Optional[Union[HSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HSConfig

    def _get_config_key(self) -> str:
        return "hs"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = HSParams(
            hm_size=int(values.get("hm_size", 20)),
            HMCR=float(values.get("HMCR", 0.9)),
            PAR=float(values.get("PAR", 0.3)),
            max_iterations=int(values.get("max_iterations", 500)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = HSSolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
