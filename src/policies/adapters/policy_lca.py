"""
LCA (League Championship Algorithm) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.lca import LCAConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.league_championship_algorithm.params import LCAParams
from policies.league_championship_algorithm.solver import LCASolver

from .factory import PolicyRegistry


@PolicyRegistry.register("lca")
class LCAPolicy(BaseBuildPolicy):
    """League Championship Algorithm policy adapter."""

    def __init__(self, config: Optional[Union[LCAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return LCAConfig

    def _get_config_key(self) -> str:
        return "lca"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = LCAParams(
            n_teams=int(values.get("n_teams", 10)),
            crossover_prob=float(values.get("crossover_prob", 0.5)),
            max_iterations=int(values.get("max_iterations", 100)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = LCASolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
