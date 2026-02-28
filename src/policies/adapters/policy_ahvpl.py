"""
AHVPL (Augmented Hybrid Volleyball Premier League) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.ahvpl import AHVPLConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.augmented_hybrid_volleyball_premier_league.ahvpl import AHVPLSolver
from policies.augmented_hybrid_volleyball_premier_league.params import AHVPLParams

from .factory import PolicyRegistry


@PolicyRegistry.register("ahvpl")
class AHVPLPolicy(BaseBuildPolicy):
    """AHVPL policy adapter."""

    def __init__(self, config: Optional[Union[AHVPLConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return AHVPLConfig

    def _get_config_key(self) -> str:
        return "ahvpl"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        # AHVPLParams nested config structure
        params = AHVPLParams.from_dict(values)

        solver = AHVPLSolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
