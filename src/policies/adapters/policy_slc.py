"""
SLC (Soccer League Competition) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.slc import SLCConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.soccer_league_competition.params import SLCParams
from policies.soccer_league_competition.solver import SLCSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("slc")
class SLCPolicy(BaseBuildPolicy):
    """Soccer League Competition policy adapter."""

    def __init__(self, config: Optional[Union[SLCConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return SLCConfig

    def _get_config_key(self) -> str:
        return "slc"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = SLCParams(
            n_teams=int(values.get("n_teams", 4)),
            team_size=int(values.get("team_size", 5)),
            stagnation_limit=int(values.get("stagnation_limit", 10)),
            max_iterations=int(values.get("max_iterations", 100)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = SLCSolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
