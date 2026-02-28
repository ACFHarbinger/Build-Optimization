"""
HGS-ALNS Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.hgs import HGSConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.hgs_alns import HGSALNSSolver
from policies.hybrid_genetic_search.params import HGSParams

from .factory import PolicyRegistry


@PolicyRegistry.register("hgs_alns")
class HGSALNSPolicy(BaseBuildPolicy):
    """HGS-ALNS policy adapter."""

    def __init__(self, config: Optional[Union[HGSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HGSConfig

    def _get_config_key(self) -> str:
        return "hgs_alns"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = HGSParams(
            population_size=int(values.get("population_size", 25)),
            elite_size=int(values.get("elite_size", 10)),
            n_offspring=int(values.get("n_offspring", 10)),
            mutation_rate=float(values.get("mutation_rate", 0.1)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = HGSALNSSolver(
            problem=problem,
            budget=budget,
            params=params,
            alns_education_iterations=int(values.get("alns_iter", 50)),
        )

        return solver.solve()
