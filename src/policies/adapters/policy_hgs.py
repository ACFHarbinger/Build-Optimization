"""
HGS (Hybrid Genetic Search) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.hgs import HGSConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.hybrid_genetic_search.hgs import HGSSolver
from policies.hybrid_genetic_search.params import HGSParams

from .factory import PolicyRegistry


@PolicyRegistry.register("hgs")
class HGSPolicy(BaseBuildPolicy):
    """HGS policy adapter."""

    def __init__(self, config: Optional[Union[HGSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HGSConfig

    def _get_config_key(self) -> str:
        return "hgs"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = HGSParams.from_dict(values)

        solver = HGSSolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
