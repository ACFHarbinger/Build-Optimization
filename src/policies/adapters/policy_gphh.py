"""
GPHH Policy Adapter.

Adapts the Genetic Programming Hyper-Heuristic (GPHH) solver to the
agnostic BaseRoutingPolicy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.gphh import GPHHConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.genetic_programming_hyper_heuristic.params import GPHHParams
from logic.src.policies.genetic_programming_hyper_heuristic.solver import GPHHSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("gphh")
class GPHHPolicy(BaseRoutingPolicy):
    """
    GPHH policy class.

    Visits bins using Genetic Programming Hyper-Heuristics — evolving
    LLH selection policies rather than routing solutions directly.
    """

    def __init__(self, config: Optional[Union[GPHHConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return GPHHConfig

    def _get_config_key(self) -> str:
        return "gphh"

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
        params = GPHHParams(
            gp_pop_size=int(values.get("gp_pop_size", 20)),
            max_gp_generations=int(values.get("max_gp_generations", 30)),
            eval_steps=int(values.get("eval_steps", 50)),
            apply_steps=int(values.get("apply_steps", 200)),
            tree_depth=int(values.get("tree_depth", 3)),
            tournament_size=int(values.get("tournament_size", 3)),
            n_llh=int(values.get("n_llh", 5)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = GPHHSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        routes, profit, cost = solver.solve()
        return routes, profit, cost
