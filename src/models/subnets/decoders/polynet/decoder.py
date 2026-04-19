"""PolyNet Decoder."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.constants.models import NUMERICAL_EPSILON
from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.subnets.decoders.common import AttentionDecoderCache
from logic.src.models.subnets.embeddings import CONTEXT_EMBEDDING_REGISTRY
from logic.src.models.subnets.embeddings.dynamic import DynamicEmbedding
from logic.src.models.subnets.modules.polynet_attention import PolyNetAttention


class PolyNetDecoder(nn.Module):
    """
    PolyNet Decoder for constructing diverse solutions.
    """

    def __init__(
        self,
        k: int,
        encoder_type: str = "AM",
        embed_dim: int = 128,
        poly_layer_dim: int = 256,
        num_heads: int = 8,
        env_name: str = "vrpp",
        mask_inner: bool = True,
        out_bias: bool = False,
        linear_bias: bool = False,
        use_graph_context: bool = True,
        check_nan: bool = True,
    ) -> None:
        """
        Initialize PolyNet decoder.
        """
        super().__init__()

        self.k = k
        self.encoder_type = encoder_type
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.env_name = env_name
        self.use_graph_context = use_graph_context

        # Context embedding
        if env_name in CONTEXT_EMBEDDING_REGISTRY:
            self.context_embedding: Union[nn.Module, nn.Linear] = CONTEXT_EMBEDDING_REGISTRY[env_name](  # type: ignore[abstract]
                embed_dim=embed_dim
            )
        else:
            self.context_embedding = nn.Linear(embed_dim, embed_dim)

        # Dynamic embedding for step context
        self.dynamic_embedding = DynamicEmbedding(embed_dim=embed_dim)

        # PolyNet pointer attention
        self.pointer = PolyNetAttention(
            k=k,
            embed_dim=embed_dim,
            poly_layer_dim=poly_layer_dim,
            num_heads=num_heads,
            mask_inner=mask_inner,
            out_bias=out_bias,
            check_nan=check_nan,
        )

        # Projections
        self.project_node_embeddings = nn.Linear(embed_dim, 3 * embed_dim, bias=linear_bias)
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=linear_bias)

    def forward(
        self,
        td: TensorDict,
        embeddings: torch.Tensor,
        env: RL4COEnvBase,
        strategy: str = "sampling",
        num_starts: int = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode solution autoregressively.
        """
        # Precompute fixed projections
        cache = self._precompute_cache(embeddings)

        outputs = []
        actions = []

        while not td["done"].all():
            # Get step logits
            logits, mask = self._get_step_logits(cache, td)

            # Convert to probabilities
            logits = logits.masked_fill(~mask, float("-inf"))
            probs = torch.softmax(logits, dim=-1)

            # Select action
            if strategy == "greedy":
                action = probs.argmax(dim=-1)
            else:
                probs = probs.clamp(min=NUMERICAL_EPSILON)
                action = torch.multinomial(probs, 1).squeeze(-1)

            # Store outputs
            log_p = torch.log(probs.gather(-1, action.unsqueeze(-1)) + NUMERICAL_EPSILON).squeeze(-1)
            outputs.append(log_p)
            actions.append(action)

            # Step environment
            td.set("action", action)
            td = env.step(td)["next"]

        # Stack outputs
        log_likelihood = torch.stack(outputs, dim=1).sum(dim=1)
        actions = torch.stack(actions, dim=1)  # type: ignore[assignment]

        return log_likelihood, actions  # type: ignore[return-value]

    def _precompute_cache(
        self,
        embeddings: torch.Tensor,
    ) -> AttentionDecoderCache:
        """Precompute fixed projections for decoding."""
        # Handle MatNet-style embeddings (tuple)
        if isinstance(embeddings, tuple):
            col_emb, row_emb = embeddings
            node_embeddings = row_emb
            context_emb = col_emb
        else:
            node_embeddings = embeddings
            context_emb = embeddings

        # Project node embeddings
        projected = self.project_node_embeddings(context_emb)
        glimpse_key, glimpse_val, logit_key = projected.chunk(3, dim=-1)

        # Graph context
        if self.use_graph_context:
            graph_context = self.project_fixed_context(context_emb.mean(dim=1))
        else:
            graph_context = torch.zeros(context_emb.size(0), self.embed_dim, device=context_emb.device)

        return AttentionDecoderCache(
            node_embeddings=node_embeddings,
            graph_context=graph_context,
            glimpse_key=glimpse_key,
            glimpse_val=glimpse_val,
            logit_key=logit_key,
        )

    def _get_step_logits(
        self,
        cache: AttentionDecoderCache,
        td: TensorDict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get logits for current step."""
        # Step context from current state
        if hasattr(self.context_embedding, "forward"):
            step_context = self.context_embedding(cache.node_embeddings, td)
        else:
            step_context = self.context_embedding(cache.node_embeddings.mean(dim=1))

        # Query = graph context + step context
        query = cache.graph_context + step_context
        query = query.unsqueeze(1)  # (batch, 1, embed_dim)

        # Dynamic embedding contribution
        glimpse_key_dyn, glimpse_val_dyn, logit_key_dyn = self.dynamic_embedding(td)

        glimpse_key = cache.glimpse_key + glimpse_key_dyn.squeeze(1)
        glimpse_val = cache.glimpse_val + glimpse_val_dyn.squeeze(1)
        logit_key = cache.logit_key + logit_key_dyn.squeeze(1)

        # Get mask
        mask = td.get("action_mask", None)

        # Compute logits via PolyNet attention
        logits = self.pointer(
            query=query,
            key=glimpse_key,
            value=glimpse_val,
            logit_key=logit_key.unsqueeze(-2),
            attn_mask=mask,
        )

        return logits.squeeze(1), mask
