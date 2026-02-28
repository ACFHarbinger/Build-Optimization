"""
HMM-GD (Hidden Markov Model Great Deluge) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.hmm_gd import HMMGDConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.hidden_markov_model_great_deluge.params import HMMGDParams
from policies.hidden_markov_model_great_deluge.solver import HMMGDSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("hmm_gd")
class HMMGDPolicy(BaseBuildPolicy):
    """HMM-GD policy adapter."""

    def __init__(self, config: Optional[Union[HMMGDConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HMMGDConfig

    def _get_config_key(self) -> str:
        return "hmm_gd"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = HMMGDParams(
            max_iterations=int(values.get("max_iterations", 1000)),
            rain_speed=float(values.get("rain_speed", 0.0001)),
            flood_margin=float(values.get("flood_margin", 0.1)),
            learning_rate=float(values.get("learning_rate", 0.1)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = HMMGDSolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
