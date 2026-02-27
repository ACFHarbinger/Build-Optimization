"""
GA (Genetic Algorithm) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ga import GAConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.genetic_algorithm.params import GAParams
from logic.src.policies.genetic_algorithm.solver import GASolver

from .factory import PolicyRegistry


@PolicyRegistry.register("ga")
class GAPolicy(BaseRoutingPolicy):
    """Genetic Algorithm policy class."""

    def __init__(self, config: Optional[Union[GAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return GAConfig

    def _get_config_key(self) -> str:
        return "ga"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        params = GAParams(
            pop_size=int(values.get("pop_size", 30)),
            max_generations=int(values.get("max_generations", 100)),
            crossover_rate=float(values.get("crossover_rate", 0.8)),
            mutation_rate=float(values.get("mutation_rate", 0.1)),
            tournament_size=int(values.get("tournament_size", 3)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = GASolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()
