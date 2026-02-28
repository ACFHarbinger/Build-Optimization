"""
RRT (Record-to-Record Travel) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.rrt import RRConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.record_to_record_travel.params import RRParams
from policies.record_to_record_travel.solver import RRSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("rrt")
class RRTPolicy(BaseBuildPolicy):
    """RRT policy adapter."""

    def __init__(self, config: Optional[Union[RRConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return RRConfig

    def _get_config_key(self) -> str:
        return "rrt"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        # Values may come from a config named 'rrt' but the params class uses 'RRParams'
        params = RRParams(
            max_iterations=int(values.get("max_iterations", 1000)),
            tolerance=float(values.get("tolerance", 0.05)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = RRSolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
