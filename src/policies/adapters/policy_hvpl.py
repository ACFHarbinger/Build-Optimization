"""
HVPL (Hybrid Volleyball Premier League) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.hvpl import HVPLConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.hybrid_volleyball_premier_league.hvpl import HVPLSolver
from policies.hybrid_volleyball_premier_league.params import HVPLParams

from .factory import PolicyRegistry


@PolicyRegistry.register("hvpl")
class HVPLPolicy(BaseBuildPolicy):
    """HVPL policy adapter."""

    def __init__(self, config: Optional[Union[HVPLConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HVPLConfig

    def _get_config_key(self) -> str:
        return "hvpl"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        # HVPLParams nested config structure
        params = HVPLParams.from_dict(values)

        solver = HVPLSolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
