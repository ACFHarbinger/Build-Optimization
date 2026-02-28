"""
LAHC (Late Acceptance Hill-Climbing) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.lahc import LAHCConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.late_acceptance_hill_climbing.params import LAHCParams
from policies.late_acceptance_hill_climbing.solver import LAHCSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("lahc")
class LAHCPolicy(BaseBuildPolicy):
    """Late Acceptance Hill-Climbing policy adapter."""

    def __init__(self, config: Optional[Union[LAHCConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return LAHCConfig

    def _get_config_key(self) -> str:
        return "lahc"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = LAHCParams(
            queue_size=int(values.get("queue_size", 50)),
            max_iterations=int(values.get("max_iterations", 1000)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = LAHCSolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
