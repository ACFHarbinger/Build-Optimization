"""
ALNS Policy Adapter.

Adapts the Adaptive Large Neighborhood Search (ALNS) logic to the agnostic interface.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies import ALNSConfig
from core.problem import BuildProblem
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.adaptive_large_neighborhood_search.alns import run_alns

from .factory import PolicyRegistry


@PolicyRegistry.register("alns")
class ALNSPolicy(BaseBuildPolicy):
    """
    ALNS policy class.
    Uses Adaptive Large Neighborhood Search for Build Optimization.
    """

    def __init__(self, config: Optional[Union[ALNSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return ALNSConfig

    def _get_config_key(self) -> str:
        return "alns"

    def _run_solver(
        self,
        problem: BuildProblem,
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        """
        Run ALNS solver.

        Returns:
            Tuple of (build_arr, score)
        """
        build_arr, score = run_alns(
            problem,
            budget,
            values,
        )
        return build_arr, score
