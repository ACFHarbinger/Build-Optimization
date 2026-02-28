"""
GLS (Guided Local Search) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.gls import GLSConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.guided_local_search.params import GLSParams
from policies.guided_local_search.solver import GLSSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("gls")
class GLSPolicy(BaseBuildPolicy):
    """Guided Local Search policy adapter."""

    def __init__(self, config: Optional[Union[GLSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return GLSConfig

    def _get_config_key(self) -> str:
        return "gls"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = GLSParams(
            lambda_param=float(values.get("lambda_param", 0.3)),
            max_restarts=int(values.get("max_restarts", 50)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 3)),
            inner_iterations=int(values.get("inner_iterations", 20)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = GLSSolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
