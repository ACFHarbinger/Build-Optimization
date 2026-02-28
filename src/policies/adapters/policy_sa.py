"""
SA (Simulated Annealing) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.sa import SAConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.simulated_annealing.params import SAParams
from policies.simulated_annealing.solver import SASolver

from .factory import PolicyRegistry


@PolicyRegistry.register("sa")
class SAPolicy(BaseBuildPolicy):
    """Simulated Annealing policy adapter."""

    def __init__(self, config: Optional[Union[SAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return SAConfig

    def _get_config_key(self) -> str:
        return "sa"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = SAParams(
            initial_temp=float(values.get("initial_temp", 100.0)),
            alpha=float(values.get("alpha", 0.995)),
            min_temp=float(values.get("min_temp", 0.01)),
            max_iterations=int(values.get("max_iterations", 500)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = SASolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
