"""
BCP (Branch-Cut-and-Price) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.bcp import BCPConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.branch_cut_and_price.dispatcher import run_bcp

from .factory import PolicyRegistry


@PolicyRegistry.register("bcp")
class BCPPolicy(BaseBuildPolicy):
    """BCP policy adapter."""

    def __init__(self, config: Optional[Union[BCPConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return BCPConfig

    def _get_config_key(self) -> str:
        return "bcp"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        # Dispatch to the BCP engines
        return run_bcp(
            problem=problem,
            budget=budget,
            values=values,
            **kwargs,
        )
