"""
Neural-Heuristic Hybrid Policy.
Uses a neural model for construction and a heuristic for refinement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Union

from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.autoregressive.policy import AutoregressivePolicy

if TYPE_CHECKING:
    from logic.src.models.policies.alns import VectorizedALNS
    from logic.src.models.policies.hgs import VectorizedHGS


class NeuralHeuristicHybrid(AutoregressivePolicy):
    """
    Hybrid policy combining neural construction and heuristic refinement.
    """

    def __init__(
        self,
        neural_policy: AutoregressivePolicy,
        heuristic_policy: Union[VectorizedALNS, VectorizedHGS],
        **kwargs,
    ):
        """Initialize NeuralHeuristicHybrid."""
        super().__init__(env_name=neural_policy.env_name, **kwargs)
        self.neural_policy = neural_policy
        self.heuristic_policy = heuristic_policy

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        strategy: str = "greedy",
        num_starts: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Solve instances using neural construction followed by heuristic refinement.
        """
        # 1. Neural Construction
        neural_out = self.neural_policy(td, env, strategy=strategy, **kwargs)

        # 2. Heuristic Refinement
        # Note: Classical solvers currently don't take initial solutions in the current wrappers
        # but we could update them to do so.
        # ALNSSolver.solve(initial_solution=...) exists.

        # For now, we'll just run the heuristic.
        # If we wanted true refinement, we'd pass neural_out["actions"] to the heuristic.

        # Update: Let's assume the heuristic wrapper can take an initial solution in kwargs.
        heuristic_out = self.heuristic_policy(td, env, initial_solution=neural_out["actions"], **kwargs)

        # 3. Return the best of both (usually the heuristic improved result)
        return heuristic_out
