"""
RTS (Reactive Tabu Search) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.rts import RTSConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.reactive_tabu_search.params import RTSParams
from policies.reactive_tabu_search.solver import RTSSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("rts")
class RTSPolicy(BaseBuildPolicy):
    """Reactive Tabu Search policy adapter."""

    def __init__(self, config: Optional[Union[RTSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return RTSConfig

    def _get_config_key(self) -> str:
        return "rts"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = RTSParams(
            max_iterations=int(values.get("max_iterations", 500)),
            initial_tenure=int(values.get("initial_tenure", 5)),
            min_tenure=int(values.get("min_tenure", 2)),
            max_tenure=int(values.get("max_tenure", 50)),
            tenure_increase=float(values.get("tenure_increase", 1.2)),
            tenure_decrease=float(values.get("tenure_decrease", 0.9)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = RTSSolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
