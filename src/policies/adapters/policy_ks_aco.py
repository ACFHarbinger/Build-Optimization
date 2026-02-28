"""
ACO Policy Adapter.

Adapts the K-Sparse Ant Colony Optimization solver to the common policy interface.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies import ACOConfig
from core.problem import BuildProblem
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.ant_colony_optimization.k_sparse_aco import run_k_sparse_aco

from .factory import PolicyRegistry


@PolicyRegistry.register("aco")
class ACOPolicy(BaseBuildPolicy):
    """
    K-Sparse Ant Colony Optimization policy class for Build Optimization.
    """

    def __init__(self, config: Optional[Union[ACOConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return ACOConfig

    def _get_config_key(self) -> str:
        return "aco"

    def _run_solver(
        self,
        problem: BuildProblem,
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        """
        Run K-Sparse ACO solver for builds.

        Returns:
            Tuple of (build_array, score)
        """
        build_arr, score = run_k_sparse_aco(
            problem,
            budget,
            values,
        )
        return build_arr, score
