"""
GA (Genetic Algorithm) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.ga import GAConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.genetic_algorithm.params import GAParams
from policies.genetic_algorithm.solver import GASolver

from .factory import PolicyRegistry


@PolicyRegistry.register("ga")
class GAPolicy(BaseBuildPolicy):
    """Genetic Algorithm policy adapter."""

    def __init__(self, config: Optional[Union[GAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return GAConfig

    def _get_config_key(self) -> str:
        return "ga"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = GAParams(
            pop_size=int(values.get("pop_size", 50)),
            max_generations=int(values.get("max_generations", 100)),
            tournament_size=int(values.get("tournament_size", 3)),
            crossover_rate=float(values.get("crossover_rate", 0.8)),
            mutation_rate=float(values.get("mutation_rate", 0.1)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = GASolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
