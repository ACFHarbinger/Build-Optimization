"""
Hyper-ACO (Hyper-Heuristic ACO) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies import ACOConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco import HyperHeuristicACO
from policies.ant_colony_optimization.hyper_heuristic_aco.params import HyperACOParams

from .factory import PolicyRegistry


@PolicyRegistry.register("hyper_aco")
class HyperACOPolicy(BaseBuildPolicy):
    """Hyper-Heuristic ACO policy adapter."""

    def __init__(self, config: Optional[Union[ACOConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return ACOConfig

    def _get_config_key(self) -> str:
        return "hyper_aco"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = HyperACOParams.from_dict(values)

        solver = HyperHeuristicACO(
            problem=problem,
            budget=budget,
            params=params,
        )

        initial_build = problem.greedy_solution()
        return solver.solve(initial_build)
