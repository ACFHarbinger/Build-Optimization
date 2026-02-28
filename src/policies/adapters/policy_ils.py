"""
ILS (Iterated Local Search) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.ils import ILSConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.iterated_local_search.params import ILSParams
from policies.iterated_local_search.solver import ILSSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("ils")
class ILSPolicy(BaseBuildPolicy):
    """Iterated Local Search policy adapter."""

    def __init__(self, config: Optional[Union[ILSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return ILSConfig

    def _get_config_key(self) -> str:
        return "ils"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = ILSParams(
            n_restarts=int(values.get("n_restarts", 50)),
            inner_iterations=int(values.get("inner_iterations", 100)),
            perturbation_strength=float(values.get("perturbation_strength", 0.2)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = ILSSolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
