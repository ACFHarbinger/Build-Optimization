"""
ABC (Artificial Bee Colony) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.abc import ABCConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.artificial_bee_colony.params import ABCParams
from policies.artificial_bee_colony.solver import ABCSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("abc")
class ABCPolicy(BaseBuildPolicy):
    """Artificial Bee Colony policy adapter."""

    def __init__(self, config: Optional[Union[ABCConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return ABCConfig

    def _get_config_key(self) -> str:
        return "abc"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = ABCParams(
            n_sources=int(values.get("n_sources", 20)),
            max_iterations=int(values.get("max_iterations", 100)),
            limit=int(values.get("limit", 20)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = ABCSolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
