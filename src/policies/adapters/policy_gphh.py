"""
GPHH (Genetic Programming Hyper-Heuristic) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.gphh import GPHHConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.genetic_programming_hyper_heuristic.params import GPHHParams
from policies.genetic_programming_hyper_heuristic.solver import GPHHSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("gphh")
class GPHHPolicy(BaseBuildPolicy):
    """GPHH policy adapter."""

    def __init__(self, config: Optional[Union[GPHHConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return GPHHConfig

    def _get_config_key(self) -> str:
        return "gphh"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = GPHHParams(
            gp_pop_size=int(values.get("gp_pop_size", 20)),
            max_gp_generations=int(values.get("max_gp_generations", 10)),
            tree_depth=int(values.get("tree_depth", 4)),
            eval_steps=int(values.get("eval_steps", 20)),
            apply_steps=int(values.get("apply_steps", 100)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=3,  # Hardcoded pool size in GPHHSolver
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = GPHHSolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
