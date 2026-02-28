"""
FA (Firefly Algorithm) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.fa import FAConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.firefly_algorithm.params import FAParams
from policies.firefly_algorithm.solver import FASolver

from .factory import PolicyRegistry


@PolicyRegistry.register("fa")
class FAPolicy(BaseBuildPolicy):
    """Firefly Algorithm policy adapter."""

    def __init__(self, config: Optional[Union[FAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return FAConfig

    def _get_config_key(self) -> str:
        return "fa"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = FAParams(
            pop_size=int(values.get("pop_size", 20)),
            beta0=float(values.get("beta0", 1.0)),
            gamma=float(values.get("gamma", 0.1)),
            alpha_rnd=float(values.get("alpha_rnd", 0.2)),
            max_iterations=int(values.get("max_iterations", 100)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = FASolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
