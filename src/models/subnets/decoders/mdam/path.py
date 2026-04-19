"""
MDAM Path module.
"""

from typing import Optional, Tuple

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.models.subnets.decoders.common import AttentionDecoderCache
from logic.src.models.subnets.decoders.mdam.attention import compute_mdam_logits
from logic.src.models.subnets.embeddings import CONTEXT_EMBEDDING_REGISTRY


class MDAMPath(nn.Module):
    """
    Single decoding path for MDAM.
    Encapsulates path-specific parameters and computation.
    """

    def __init__(
        self,
        embed_dim: int,
        env_name: str,
        num_heads: int,
        tanh_clipping: float = 10.0,
        mask_inner: bool = True,
        mask_logits: bool = True,
    ) -> None:
        """Initialize Class.

        Args:
            embed_dim (int): Description of embed_dim.
            env_name (str): Description of env_name.
            num_heads (int): Description of num_heads.
            tanh_clipping (float): Description of tanh_clipping.
            mask_inner (bool): Description of mask_inner.
            mask_logits (bool): Description of mask_logits.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        # Context embedding
        if env_name in CONTEXT_EMBEDDING_REGISTRY:
            self.context_embedding = CONTEXT_EMBEDDING_REGISTRY[env_name](embed_dim=embed_dim)  # type: ignore[abstract]
        else:
            self.context_embedding = nn.Linear(embed_dim, embed_dim)  # type: ignore[assignment]

        # Projections
        self.project_node_embeddings = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_step_context = nn.Linear(2 * embed_dim, embed_dim, bias=False)
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=False)

    def precompute(
        self,
        h_embed: torch.Tensor,
        num_steps: int = 1,
    ) -> AttentionDecoderCache:
        """Precompute fixed projections for this path."""
        graph_embed = h_embed.mean(dim=1)

        # Fixed context projection
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # Node embedding projections
        projected = self.project_node_embeddings(h_embed[:, None, :, :])
        glimpse_key, glimpse_val, logit_key = projected.chunk(3, dim=-1)

        return AttentionDecoderCache(
            node_embeddings=h_embed,
            graph_context=fixed_context,
            glimpse_key=self._make_heads(glimpse_key, num_steps),
            glimpse_val=self._make_heads(glimpse_val, num_steps),
            logit_key=logit_key.contiguous(),
        )

    def _make_heads(
        self,
        v: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Reshape for multi-head attention."""
        # v: (batch, 1, graph_size, 3*embed_dim) or similar
        # target: (n_heads, batch, num_steps, graph_size, head_dim)

        # NOTE: v comes from chunk(3, dim=-1) of (batch, 1, graph_size, 3*embed_dim)
        # so v is (batch, 1, graph_size, embed_dim)

        batch_size = v.size(0)
        # v dim 1 is 1
        graph_size = v.size(2)

        # Reshape to (batch, 1, graph_size, num_heads, head_dim)
        v = v.view(batch_size, 1, graph_size, self.num_heads, -1)

        # Expand num_steps
        steps = num_steps if num_steps is not None else 1
        v = v.expand(batch_size, steps, graph_size, self.num_heads, -1)

        # Permute to (num_heads, batch, num_steps, graph_size, head_dim)
        return v.permute(3, 0, 1, 2, 4)

    def get_logprobs(
        self,
        fixed: AttentionDecoderCache,
        td: TensorDict,
        dynamic_embed: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        path_index: int,  # Needed for compute_mdam_logits signature?
        # Actually compute_mdam_logits takes path_index but mostly for identification?
        # Wait, compute_mdam_logits uses path_index?
        # Let's check compute_mdam_logits again.
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get logprobs.

        Args:
            fixed (AttentionDecoderCache): Description of fixed.
            td (TensorDict): Description of td.
            dynamic_embed (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Description of dynamic_embed.
            path_index (int): Description of path_index.

        Returns:
            Any: Description of return value.
        """
        # Step context
        if hasattr(self.context_embedding, "forward"):
            step_context = self.context_embedding(fixed.node_embeddings, td)
        else:
            step_context = self.context_embedding(fixed.node_embeddings.mean(dim=1))

        if step_context.dim() == 2:
            step_context = step_context.unsqueeze(1)

        glimpse_q = fixed.graph_context + step_context

        try:
            glimpse_key_dyn, glimpse_val_dyn, logit_key_dyn = dynamic_embed
        except ValueError as e:
            print(
                f"Error unpacking dynamic_embed: {len(dynamic_embed) if isinstance(dynamic_embed, tuple) else 'not tuple'}"
            )
            raise e

        glimpse_k = (fixed.glimpse_key or 0.0) + glimpse_key_dyn
        glimpse_v = (fixed.glimpse_val or 0.0) + glimpse_val_dyn
        logit_k = (fixed.logit_key or 0.0) + logit_key_dyn

        mask = td.get("action_mask", None)

        logprobs, _ = compute_mdam_logits(
            query=glimpse_q,
            glimpse_K=glimpse_k,
            glimpse_V=glimpse_v,
            logit_K=logit_k,
            mask=mask,
            num_heads=self.num_heads,
            project_out=self.project_out,
            tanh_clipping=self.tanh_clipping,
            mask_inner=self.mask_inner,
            mask_logits=self.mask_logits,
        )

        return logprobs, mask
