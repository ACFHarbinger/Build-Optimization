"""
MDAM Multi-Path Decoder implementation.
"""

from typing import Any, Tuple, Union, cast

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.subnets.decoders.common import select_action
from logic.src.models.subnets.decoders.mdam.path import MDAMPath
from logic.src.models.subnets.embeddings.dynamic import DynamicEmbedding


class MDAMDecoder(nn.Module):
    """
    MDAM Multi-Path Decoder.

    Constructs diverse solutions by running multiple decoder paths in parallel.
    Each path has its own context embedding and projection layers.
    KL divergence between path outputs encourages solution diversity.

    Reference:
        Xin et al. "Multi-Decoder Attention Model with Embedding Glimpse for
        Solving Vehicle Routing Problems" (AAAI 2021)
    """

    paths: nn.ModuleList
    W_placeholder: nn.Parameter

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_paths: int = 5,
        env_name: str = "vrpp",
        mask_inner: bool = True,
        mask_logits: bool = True,
        eg_step_gap: int = 200,
        tanh_clipping: float = 10.0,
        train_strategy: str = "sampling",
        val_strategy: str = "greedy",
        test_strategy: str = "greedy",
    ) -> None:
        """
        Initialize MDAM decoder.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            num_paths: Number of parallel decoder paths for diversity.
            env_name: Environment name for context embedding selection.
            mask_inner: Whether to mask inner attention.
            mask_logits: Whether to mask final logits.
            eg_step_gap: Gap between encoder update steps during decoding.
            tanh_clipping: Clipping value for tanh on logits.
            train_strategy: Decoding type during training.
            val_strategy: Decoding type during validation.
            test_strategy: Decoding type during testing.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_paths = num_paths
        self.env_name = env_name
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.eg_step_gap = eg_step_gap
        self.tanh_clipping = tanh_clipping

        self.train_strategy = train_strategy
        self.val_strategy = val_strategy
        self.test_strategy = test_strategy

        # Dynamic embedding for step context
        self.dynamic_embedding = DynamicEmbedding(embed_dim=embed_dim)

        # Placeholder for first step
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embed_dim))
        self.W_placeholder.data.uniform_(-1, 1)

        # Path-specific modules
        self.paths = nn.ModuleList(
            [
                MDAMPath(
                    embed_dim=embed_dim,
                    env_name=env_name,
                    num_heads=num_heads,
                    tanh_clipping=tanh_clipping,
                    mask_inner=mask_inner,
                    mask_logits=mask_logits,
                )
                for _ in range(num_paths)
            ]
        )

    def forward(
        self,
        td: TensorDict,
        embeddings: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        env: RL4COEnvBase,
        strategy: str = "greedy",
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MDAM Multi-path decoding.
        """
        # Unpack MDAM specific embeddings
        h, _, attn, V, h_old = cast(
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            embeddings,
        )
        encoder = kwargs.get("encoder")

        if strategy is None:
            strategy = self.train_strategy

        # First step: calculate KL divergence between paths
        kl_divergence = self._compute_initial_kl_divergence(td, h)

        # Main decoding for each path
        reward_list = []
        ll_list = []
        action_list = []

        for path_idx in range(self.num_paths):
            reward, ll, actions = self._decode_path(
                td.clone(),
                h.clone(),
                env,
                attn.clone(),
                V.clone(),
                h_old.clone(),
                encoder,
                path_idx,
                strategy,
            )
            reward_list.append(reward)
            ll_list.append(ll)
            action_list.append(actions)

        # Stack results
        reward = torch.stack(reward_list, dim=1)  # (batch, num_paths)
        log_likelihood = torch.stack(ll_list, dim=1)  # (batch, num_paths)

        # Return actions from first path (or could select best)
        actions = action_list[0]

        return reward, log_likelihood, kl_divergence, actions

    def _compute_initial_kl_divergence(
        self,
        td: TensorDict,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between path logprobs at first step."""
        if self.num_paths <= 1:
            return torch.tensor(0.0, device=h.device)

        output_list = []
        # We need dynamic embedding for the current step (step 0)
        dynamic_embed = self.dynamic_embedding(td)

        for path_idx in range(self.num_paths):
            path = cast(MDAMPath, self.paths[path_idx])
            fixed = path.precompute(h)
            logprobs, _ = path.get_logprobs(fixed, td, dynamic_embed, path_idx)

            # Ensure numerical stability
            logprobs = torch.clamp(logprobs, min=-1e9)
            output_list.append(logprobs[:, 0, :])

        # Compute pairwise KL divergence
        kl_divergences = []
        for i in range(self.num_paths):
            for j in range(self.num_paths):
                if i == j:
                    continue
                # KL(P_i || P_j) = sum(P_i * (log P_i - log P_j))
                kl = torch.sum(
                    torch.exp(output_list[i]) * (output_list[i] - output_list[j]),
                    dim=-1,
                )
                kl_divergences.append(kl)

        return torch.stack(kl_divergences, dim=0).mean()

    def _decode_path(
        self,
        td: TensorDict,
        h: torch.Tensor,
        env: RL4COEnvBase,
        attn: torch.Tensor,
        V: torch.Tensor,
        h_old: torch.Tensor,
        encoder: Any,
        path_idx: int,
        strategy: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode a single path."""
        outputs = []
        actions = []

        path = cast(MDAMPath, self.paths[path_idx])
        fixed = path.precompute(h)
        step = 0

        while not td["done"].all():
            # Periodic encoder update
            if step > 1 and step % self.eg_step_gap == 0:
                mask = td.get("action_mask", None)
                if mask is not None and hasattr(encoder, "change"):
                    # Update embeddings with current mask
                    h, _ = encoder.change(attn, V, h_old, mask)
                    fixed = path.precompute(h)

            # Get logprobs and mask
            dynamic_embed = self.dynamic_embedding(td)
            logprobs, mask = path.get_logprobs(fixed, td, dynamic_embed, path_idx)

            # Select action
            probs = torch.exp(logprobs[:, 0, :])
            action = select_action(probs, mask, strategy=strategy)

            # Step environment
            td.set("action", action)
            td = env.step(td)["next"]

            outputs.append(logprobs[:, 0, :])
            actions.append(action)
            step += 1

        # Compute reward and log-likelihood
        outputs = torch.stack(outputs, dim=1)  # type: ignore[assignment]  # (batch, seq_len, graph_size)
        actions = torch.stack(actions, dim=1)  # type: ignore[assignment]  # (batch, seq_len)

        reward = env.get_reward(td, actions)  # type: ignore[arg-type]
        ll = self._get_log_likelihood(outputs, actions)  # type: ignore[arg-type]

        return reward, ll, actions  # type: ignore[return-value]

    def _get_log_likelihood(
        self,
        log_probs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log-likelihood of action sequence."""
        # log_probs: (batch, seq_len, num_nodes)
        # actions: (batch, seq_len)
        ll = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        return ll.sum(dim=-1)
