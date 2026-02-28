"""
SCA (Sine Cosine Algorithm) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.sca import SCAConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.sine_cosine_algorithm.params import SCAParams
from policies.sine_cosine_algorithm.solver import SCASolver

from .factory import PolicyRegistry


@PolicyRegistry.register("sca")
class SCAPolicy(BaseBuildPolicy):
    """SCA policy adapter."""

    def __init__(self, config: Optional[Union[SCAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return SCAConfig

    def _get_config_key(self) -> str:
        return "sca"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = SCAParams(
            pop_size=int(values.get("pop_size", 20)),
            a_max=float(values.get("a_max", 2.0)),
            max_iterations=int(values.get("max_iterations", 100)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = SCASolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
