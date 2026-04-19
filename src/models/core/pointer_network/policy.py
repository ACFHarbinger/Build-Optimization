"""
Pointer Network Policy Adapter.
"""

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.autoregressive.policy import AutoregressivePolicy
from logic.src.models.core.pointer_network import PointerNetwork
from logic.src.utils.tasks.dummy_problem import DummyProblem


class PointerNetworkPolicy(AutoregressivePolicy):
    """
    Pointer Network Policy Adapter.

    Adapts the classic RNN-based Pointer Network to the new architecture.
    """

    def __init__(
        self,
        env_name: str,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        **kwargs,
    ):
        """Initialize PointerNetworkPolicy."""
        super().__init__(env_name=env_name, embed_dim=embed_dim)
        self.model = PointerNetwork(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            problem=DummyProblem(env_name),
            **kwargs,
        )

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        strategy: str = "sampling",
        num_starts: int = 1,
        actions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass using Pointer Network subnets.
        """
        batch_size, graph_size, _ = td["locs"].size()
        inputs = td["locs"]

        # PointerNetwork specific embedding and encoding
        # (Adapted from PointerNetwork.forward)
        embedded_inputs = torch.mm(
            inputs.transpose(0, 1).contiguous().view(-1, inputs.size(-1)),
            self.model.embedding,
        ).view(graph_size, batch_size, -1)

        # Set strategy in legacy decoder
        self.model.set_strategy(strategy)

        # We reuse the logic from PointerNetwork._inner but integrate with env
        # Actually, PointerNetwork._inner calls self.decoder which handles the whole loop
        # if eval_tours is provided.

        # If we want to be consistent with other policies, we would step through the env.
        # But PointerNetwork is tightly coupled with its RNN decoder.

        # Let's try to pass 'actions' as 'eval_tours' to PointerNetwork for teacher forcing
        # or just run the internal loop.

        if actions is not None:
            # Teacher forcing using legacy codes' eval_tours
            log_p_output, out_actions = self.model._inner(embedded_inputs, eval_tours=actions)
        else:
            # Sampling / Greedy
            log_p_output, out_actions = self.model._inner(embedded_inputs)

        # PointerNetwork returns log_p per step [batch, steps, nodes]
        # We need to gather the log probabilities of selected actions
        log_p = log_p_output.gather(2, out_actions.unsqueeze(-1)).squeeze(-1)

        # Calculate total log likelihood
        # We might need to mask out padded actions if any, but in basic VRP it's uniform
        log_likelihood = log_p.sum(dim=1)

        # Calculate reward using environment
        reward = env.get_reward(td, out_actions)

        return {
            "reward": reward,
            "log_likelihood": log_likelihood,
            "actions": out_actions,
        }
