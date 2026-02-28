"""
OBA (Old Bachelor Acceptance) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.oba import OBAConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.old_bachelor_acceptance.params import OBAParams
from policies.old_bachelor_acceptance.solver import OBASolver

from .factory import PolicyRegistry


@PolicyRegistry.register("oba")
class OBAPolicy(BaseBuildPolicy):
    """OBA policy adapter."""

    def __init__(self, config: Optional[Union[OBAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return OBAConfig

    def _get_config_key(self) -> str:
        return "oba"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = OBAParams(
            max_iterations=int(values.get("max_iterations", 1000)),
            contraction=float(values.get("contraction", 0.01)),
            dilation=float(values.get("dilation", 0.02)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = OBASolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
