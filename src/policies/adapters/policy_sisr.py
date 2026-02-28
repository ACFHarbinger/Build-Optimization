"""
SISR (Slack Induction by String Removal) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.sisr import SISRConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.slack_induction_by_string_removal.params import SISRParams
from policies.slack_induction_by_string_removal.solver import SISRSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("sisr")
class SISRPolicy(BaseBuildPolicy):
    """SISR policy adapter."""

    def __init__(self, config: Optional[Union[SISRConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return SISRConfig

    def _get_config_key(self) -> str:
        return "sisr"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = SISRParams(
            max_iterations=int(values.get("max_iterations", 1000)),
            destroy_ratio=float(values.get("destroy_ratio", 0.2)),
            blink_rate=float(values.get("blink_rate", 0.1)),
            start_temp=float(values.get("start_temp", 100.0)),
            cooling_rate=float(values.get("cooling_rate", 0.999)),
            max_string_len=int(values.get("max_string_len", 10)),
            avg_string_len=float(values.get("avg_string_len", 5.0)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = SISRSolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
