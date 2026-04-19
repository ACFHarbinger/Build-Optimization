"""Deep GAT Decoder module."""

from __future__ import annotations

import math
from typing import Any, Optional, cast

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.decoders.common import AttentionDecoderCache
from logic.src.models.subnets.decoders.gat.graph_decoder import GraphAttentionDecoder


class DeepGATDecoder(nn.Module):
    """
    Deep Decoder module independent of the model architecture.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        dropout_rate: float = 0.1,
        aggregation_graph: str = "avg",
        mask_graph: bool = False,
        mask_logits: bool = True,
        tanh_clipping: float = 10.0,
        seed: int = 42,
        temp: float = 1.0,
        **kwargs,
    ):
        """Initialize DeepGATDecoder."""
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.aggregation_graph = aggregation_graph
        self.mask_graph = mask_graph
        self.mask_logits = mask_logits
        self.tanh_clipping = tanh_clipping
        self.seed = seed
        device = kwargs.get("device", "cpu")
        self.generator = torch.Generator(device=device).manual_seed(self.seed)
        self.temp = temp

        self.decoder = GraphAttentionDecoder(
            n_heads=n_heads,
            embed_dim=embed_dim,
            n_layers=n_layers,
            feed_forward_hidden=hidden_dim,
            norm_config=norm_config,
            activation_config=activation_config,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        # Projections
        # Input to project_fixed_context is graph_embed (embed_dim)
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_step_context = nn.Linear(2 * embed_dim + 1, embed_dim, bias=False)

    @property
    def device(self) -> torch.device:
        """Get device of the model."""
        return next(self.parameters()).device

    def forward(
        self,
        input: torch.Tensor,
        embeddings: torch.Tensor,
        fixed_context: Optional[torch.Tensor] = None,
        init_context: Optional[torch.Tensor] = None,
        env: Optional[Any] = None,
        expert_pi: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        """Standard Module forward wrapper."""
        return self._inner(
            input,
            embeddings,
            fixed_context,
            init_context,
            env,
            expert_pi,
            **kwargs,
        )

    def _inner(
        self,
        nodes: torch.Tensor,
        embeddings: torch.Tensor,
        fixed_context: Optional[torch.Tensor] = None,
        init_context: Optional[torch.Tensor] = None,
        env: Optional[Any] = None,
        expert_pi: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        """Constructive decoding loop."""
        outputs = []
        sequences = []

        cost_weights = kwargs.get("cost_weights")
        dist_matrix = kwargs.get("dist_matrix")

        # Get problem-specific state implementation
        if hasattr(self, "problem") and self.problem is not None:
            state = cast(Any, self.problem).make_state(nodes, None, cost_weights, dist_matrix, **kwargs)
        else:
            # Fallback or error if problem is missing
            raise AttributeError("DeepGATDecoder requires 'self.problem' to be set for constructive decoding.")

        fixed = self._precompute(embeddings)

        strategy = kwargs.get("strategy", "sampling")

        # Try to get graph size for safety break
        try:
            if isinstance(nodes, torch.Tensor):
                graph_size = nodes.shape[1]
            elif hasattr(nodes, "__getitem__") and "locs" in nodes:
                graph_size = nodes["locs"].shape[1]
            else:
                graph_size = 100
        except Exception:
            graph_size = 100

        # Safety break for infinite loops (e.g. 10x graph size)
        max_steps = max(100, graph_size * 10)

        i = 0
        while not state.all_finished() and i < max_steps:
            log_p, mask = self._get_log_p(fixed, state)
            selected = self._select_node(log_p.exp(), mask, strategy=strategy)

            state = state.update(selected)

            outputs.append(log_p)
            sequences.append(selected)
            i += 1

        if i >= max_steps:
            print(f" [!] Warning: Decoding reached max_steps ({max_steps}). Possible infinite loop.")

        _log_p = torch.stack(outputs, 1)
        pi = torch.stack(sequences, 1)

        cost = None
        if hasattr(self, "problem") and hasattr(cast(Any, self.problem), "get_costs"):
            out_cost = cast(Any, self.problem).get_costs(nodes, pi, None)
            cost = out_cost[0] if isinstance(out_cost, tuple) else out_cost

        return _log_p, pi, cost, None

    def _select_node(self, probs, mask, strategy="greedy"):
        """Selection logic."""
        if strategy == "greedy":
            _, selected = probs.max(1)
        elif strategy == "sampling":
            selected = torch.multinomial(probs, 1, generator=self.generator).squeeze(1)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return selected

    def __getstate__(self):
        """Prepare state for pickling (handle non-picklable Generator)."""
        state = self.__dict__.copy()
        state["generator_state"] = self.generator.get_state()
        state["generator_device"] = str(self.generator.device)
        del state["generator"]
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        gen_state = state.pop("generator_state")
        gen_device = state.pop("generator_device")
        self.__dict__.update(state)
        self.generator = torch.Generator(device=gen_device)
        self.generator.set_state(gen_state)

    def _precompute(self, embeddings, num_steps=1):
        """Precompute fixed context for decoding."""
        if self.aggregation_graph == "avg":
            graph_embed = embeddings.mean(1)
        elif self.aggregation_graph == "sum":
            graph_embed = embeddings.sum(1)
        elif self.aggregation_graph == "max":
            graph_embed = embeddings.max(1)[0]
        else:
            graph_embed = embeddings.sum(1) * 0.0

        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        return AttentionDecoderCache(
            node_embeddings=embeddings,
            graph_context=fixed_context,
        )

    def _get_log_p(self, fixed, state, normalize=True):
        """Evaluate log probabilities for all nodes at once."""
        step_context = self._get_parallel_step_context(fixed.node_embeddings, state)

        query = fixed.context_node_projected + self.project_step_context(step_context)

        # Pass raw embeddings to decoder
        mha_K = fixed.node_embeddings

        mask = state.get_mask()
        graph_mask = state.get_edges_mask() if self.mask_graph else None

        log_p = self._one_to_many_logits(query, mha_K, mask, graph_mask)
        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)
        return log_p, mask

    def _one_to_many_logits(self, query, mha_K, mask, graph_mask):
        """Interact with GATDecoder layers directly to refine query."""
        q = query
        for layer in self.decoder.layers:
            q = layer(q, mha_K, mask)

        # Compute proper pointer attention logits: (B, 1, Nodes)
        logits = torch.matmul(q, mha_K.transpose(-2, -1)) / math.sqrt(self.embed_dim)

        if self.mask_logits and self.mask_graph:
            logits[graph_mask] = -math.inf

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        if self.mask_logits:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            # Apply mask to logits
            logits[mask] = -math.inf

        return logits

    def _get_parallel_step_context(self, embeddings, state):
        """Get step context for all batches in parallel."""
        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()

        if num_steps > 1:
            raise NotImplementedError("Parallel steps > 1 not supported yet")

        current_node_embed = torch.gather(
            embeddings,
            1,
            current_node.unsqueeze(-1).expand(-1, -1, embeddings.size(-1)),
        ).view(batch_size, num_steps, -1)

        return torch.cat(
            [
                current_node_embed,
                torch.zeros_like(current_node_embed),  # Placeholder for graph context/helpers
                torch.zeros(batch_size, num_steps, 1, device=embeddings.device),
            ],
            dim=-1,
        )
