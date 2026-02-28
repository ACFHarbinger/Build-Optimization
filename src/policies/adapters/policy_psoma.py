"""
PSOMA (Particle Swarm Optimization Memetic) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.psoma import PSOMAConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.particle_swarm_optimization_memetic.params import PSOMAParams
from policies.particle_swarm_optimization_memetic.solver import PSOMAsSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("psoma")
class PSOMAPolicy(BaseBuildPolicy):
    """PSOMA policy adapter."""

    def __init__(self, config: Optional[Union[PSOMAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return PSOMAConfig

    def _get_config_key(self) -> str:
        return "psoma"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = PSOMAParams(
            pop_size=int(values.get("pop_size", 20)),
            omega=float(values.get("omega", 0.1)),
            c1=float(values.get("c1", 1.5)),
            c2=float(values.get("c2", 2.0)),
            max_iterations=int(values.get("max_iterations", 200)),
            local_search_freq=int(values.get("local_search_freq", 10)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = PSOMAsSolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
