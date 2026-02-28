"""
QDE (Quantum-Inspired Differential Evolution) Policy Adapter.
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

from configs.policies.qde import QDEConfig
from policies.adapters.base_build_policy import BaseBuildPolicy
from policies.quantum_differential_evolution.params import QDEParams
from policies.quantum_differential_evolution.solver import QDESolver

from .factory import PolicyRegistry


@PolicyRegistry.register("qde")
class QDEPolicy(BaseBuildPolicy):
    """QDE policy adapter."""

    def __init__(self, config: Optional[Union[QDEConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return QDEConfig

    def _get_config_key(self) -> str:
        return "qde"

    def _run_solver(
        self,
        problem: Any,  # BuildProblem
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        params = QDEParams(
            pop_size=int(values.get("pop_size", 20)),
            F=float(values.get("F", 0.5)),
            CR=float(values.get("CR", 0.9)),
            max_iterations=int(values.get("max_iterations", 100)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = QDESolver(
            problem=problem,
            budget=budget,
            params=params,
        )

        return solver.solve()
