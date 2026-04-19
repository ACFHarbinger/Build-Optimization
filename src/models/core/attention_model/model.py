"""
Main AttentionModel class assembling all mixins.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

import torch
import torch.utils.checkpoint
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.constants.models import (
    FEED_FORWARD_EXPANSION,
    NODE_DIM,
    TANH_CLIPPING,
)
from logic.src.interfaces.tensor_dict_like import ITensorDictLike
from logic.src.models.subnets.embeddings import (
    GenericContextEmbedder,
    VRPPContextEmbedder,
    WCVRPContextEmbedder,
)
from logic.src.models.subnets.factories import NeuralComponentFactory
from logic.src.utils.functions.problem import is_tsp_problem, is_vrpp_problem, is_wc_problem

from .decoding import DecodingMixin


class AttentionModel(DecodingMixin, nn.Module):
    """
    Attention Model for Vehicle Routing Problems.

    This model uses an Encoder-Decoder architecture with Multi-Head Attention to solve
    various VRP instances (VRPP, WCVRP, CWCVRP). It encodes the problem graph and
    constructively decodes the solution one step at a time.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        problem: Any,
        component_factory: NeuralComponentFactory,
        n_encode_layers: int = 2,
        n_encode_sublayers: Optional[int] = None,
        n_decode_layers: Optional[int] = None,
        dropout_rate: float = 0.1,
        aggregation: str = "sum",
        aggregation_graph: str = "avg",
        tanh_clipping: float = TANH_CLIPPING,
        mask_inner: bool = True,
        mask_logits: bool = True,
        mask_graph: bool = False,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        n_heads: int = 8,
        checkpoint_encoder: bool = False,
        shrink_size: Optional[int] = None,
        pomo_size: int = 0,
        temporal_horizon: int = 0,
        spatial_bias: bool = False,
        spatial_bias_scale: float = 1.0,
        entropy_weight: float = 0.0,
        predictor_layers: Optional[int] = None,
        connection_type: str = "residual",
        hyper_expansion: int = FEED_FORWARD_EXPANSION,
        decoder_type: str = "attention",
        **kwargs,
    ) -> None:
        """
        Initialize the Attention Model.
        """
        nn.Module.__init__(self)
        DecodingMixin.__init__(self)

        if norm_config is None:
            norm_config = NormalizationConfig()

        if activation_config is None:
            activation_config = ActivationConfig()

        self._init_parameters(
            embed_dim,
            hidden_dim,
            problem,
            n_heads,
            pomo_size,
            checkpoint_encoder,
            aggregation_graph,
            temporal_horizon,
            tanh_clipping,
        )

        self._init_context_embedder(temporal_horizon)

        # Make step context dim dependent on problem/decoder
        step_context_dim = 2 * embed_dim + (embed_dim if is_tsp_problem(problem) else embed_dim)
        # Note: Logic for step_context_dim can be more complex depending on the specific problem
        # and decoder implementation, but this serves as a baseline.

        self._init_components(
            component_factory,
            step_context_dim,
            n_encode_layers,
            n_encode_sublayers,
            n_decode_layers,
            norm_config,
            activation_config,
            dropout_rate,
            aggregation,
            hyper_expansion,
            connection_type,
            predictor_layers,
            tanh_clipping,
            mask_inner,
            mask_logits,
            mask_graph,
            shrink_size,
            spatial_bias,
            spatial_bias_scale,
            decoder_type,
        )

        self.set_strategy("greedy")

    def _init_parameters(
        self,
        embed_dim: int,
        hidden_dim: int,
        problem: Any,
        n_heads: int,
        pomo_size: int,
        checkpoint_encoder: bool,
        aggregation_graph: str,
        temporal_horizon: int,
        tanh_clipping: float,
    ):
        """Initialize basic model parameters."""
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.problem = problem
        self.pomo_size = pomo_size
        self.checkpoint_encoder = checkpoint_encoder
        self.aggregation_graph = aggregation_graph
        self.temporal_horizon = temporal_horizon
        self.tanh_clipping = tanh_clipping

    def _init_context_embedder(self, temporal_horizon: int):
        """Initialize the context embedder strategy."""
        if is_vrpp_problem(self.problem):
            self.context_embedder = VRPPContextEmbedder(self.embed_dim, temporal_horizon=self.temporal_horizon)
        elif is_wc_problem(self.problem):
            self.context_embedder = WCVRPContextEmbedder(self.embed_dim, temporal_horizon=self.temporal_horizon)  # type: ignore[assignment]
        else:
            node_dim = 2 if is_tsp_problem(self.problem) else NODE_DIM
            self.context_embedder = GenericContextEmbedder(  # type: ignore[assignment]
                self.embed_dim,
                node_dim=node_dim,
                temporal_horizon=self.temporal_horizon,
            )

    @property
    def is_vrpp(self):
        """Is vrpp.

        Returns:
            Any: Description.
        """
        return is_vrpp_problem(self.problem)

    @property
    def is_wc(self):
        """Is wc.

        Returns:
            Any: Description.
        """
        return is_wc_problem(self.problem)

    def _init_components(
        self,
        component_factory: NeuralComponentFactory,
        step_context_dim: int,
        n_encode_layers: int,
        n_encode_sublayers: Optional[int],
        n_decode_layers: Optional[int],
        norm_config: NormalizationConfig,
        activation_config: ActivationConfig,
        dropout_rate: float,
        aggregation: str,
        hyper_expansion: int,
        connection_type: str,
        predictor_layers: Optional[int],
        tanh_clipping: float,
        mask_inner: bool,
        mask_logits: bool,
        mask_graph: bool,
        shrink_size: Optional[int],
        spatial_bias: bool,
        spatial_bias_scale: float,
        decoder_type: str = "attention",
    ):
        """Initialize encoder and decoder components using the factory."""
        # Encoder
        self.encoder = component_factory.create_encoder(
            embed_dim=self.embed_dim,
            n_layers=n_encode_layers,
            n_sublayers=n_encode_sublayers,
            norm_config=norm_config,
            activation_config=activation_config,
            dropout_rate=dropout_rate,
            aggregation=aggregation,
            hyper_expansion=hyper_expansion,
            connection_type=connection_type,
            n_heads=self.n_heads,
            mask_inner=mask_inner,
            mask_graph=mask_graph,
            spatial_bias=spatial_bias,
            spatial_bias_scale=spatial_bias_scale,
        )

        # Decoder
        self.decoder = component_factory.create_decoder(
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            problem=self.problem,
            n_layers=n_decode_layers,
            n_heads=self.n_heads,
            step_context_dim=step_context_dim,
            predictor_layers=predictor_layers,
            norm_config=norm_config,
            activation_config=activation_config,
            dropout_rate=dropout_rate,
            aggregation=aggregation,
            hyper_expansion=hyper_expansion,
            connection_type=connection_type,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            shrink_size=shrink_size,
            decoder_type=decoder_type,
        )

    def _get_initial_embeddings(self, input: Dict[str, torch.Tensor]):
        """
        Get initial node embeddings from the context embedder.

        Args:
            input: The input data dictionary.

        Returns:
            Tuple containing (initial node embeddings, initial context).
        """
        return self.context_embedder(input), None

    def forward(
        self,
        input: Dict[str, torch.Tensor],
        env: Optional[Any] = None,
        strategy: Optional[str] = None,
        return_pi: bool = False,
        pad: bool = False,
        mask: Optional[torch.Tensor] = None,
        expert_pi: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Forward pass of the Attention Model.

        Args:
            input: Problem state dictionary/TensorDict.
            env: Environment instance.
            strategy: Decoding strategy ('greedy' or 'sampling').
            return_pi: Whether to return the policy log probabilities.
            pad: Whether to pad keys for non-square matrices (unused here, kept for API compat).
            mask: Optional mask for valid nodes.
            expert_pi: Optional expert policy for imitation learning.
            **kwargs: Additional arguments.

        Returns:
            Dictionary containing 'reward', 'cost', 'log_likelihood', and optionally 'pi'.
        """
        # Ensure embedding init
        if not hasattr(self, "project_node_embeddings"):
            # Should have been called in __init__, but safe guard
            pass

        # Use efficient embeddings
        embeddings, init_context = self._get_initial_embeddings(input)

        # Pass through Encoder
        if self.checkpoint_encoder and self.training:
            # Gradient checkpointing for memory efficiency
            outputs = cast(
                torch.Tensor, torch.utils.checkpoint.checkpoint(self.encoder, embeddings, mask, use_reentrant=False)
            )
        else:
            outputs = self.encoder(embeddings, mask)

        # Graph aggregation for context
        graph_context = self._aggregate_graph_context(outputs)

        # Pass through Decoder
        # The decoder handles autoregressive steps
        out_dec = self.decoder(
            input,
            outputs,
            graph_context,
            init_context,
            env,
            strategy=strategy or self.strategy,
            return_pi=return_pi,
            expert_pi=expert_pi,
        )

        # Handle varying return counts from legacy / experimental decoders
        if isinstance(out_dec, tuple):
            if len(out_dec) == 4:
                _log_p, pi, cost, final_td = out_dec
            elif len(out_dec) == 3:
                _log_p, pi, cost = out_dec
                final_td = None
            else:
                _log_p = out_dec[0]
                pi = out_dec[1] if len(out_dec) > 1 else None
                cost = out_dec[2] if len(out_dec) > 2 else None
                final_td = out_dec[3] if len(out_dec) > 3 else None
        else:
            _log_p = out_dec
            pi = None
            cost = None
            final_td = None

        reward = -cost if cost is not None else torch.tensor(0.0, device=outputs.device)
        out = {"cost": cost, "reward": reward, "td": final_td}  # RL maximizes reward

        if _log_p is not None:
            out["log_likelihood"] = _log_p
            out["log_p"] = _log_p

        if pi is not None:
            out["actions"] = pi
            if return_pi:
                out["pi"] = pi

        return out

    def _aggregate_graph_context(self, outputs: torch.Tensor) -> torch.Tensor:
        """Helper to aggregate node embeddings into a graph-level context."""
        if self.aggregation_graph == "avg":
            return outputs.mean(dim=1)
        if self.aggregation_graph == "max":
            return outputs.max(dim=1)[0]
        if self.aggregation_graph == "sum":
            return outputs.sum(dim=1)
        return outputs.mean(dim=1)

    def precompute_fixed(self, input: Dict[str, torch.Tensor], edges: Optional[torch.Tensor]):
        """
        Precompute fixed embeddings for the input.

        Args:
            input: The input data.
            edges: Edge information for the graph.

        Returns:
            CachedLookup: A cached lookup object containing precomputed decoder state.
        """
        from logic.src.utils.decoding import CachedLookup

        embeddings, init_context = self._get_initial_embeddings(input)
        out = self.decoder(input, embeddings, init_context, None, precompute_only=True)

        # Handle varying return counts from legacy / experimental decoders
        if isinstance(out, tuple):
            if len(out) >= 3:
                _log_p, _pi, _cost = out[:3]
            else:
                _log_p = out[0]
        else:
            _log_p = out

        # Return a lookup object compatible with beam search
        # Note: The exact structure depends on what beam_search expects
        # This is a simplified placeholder based on typical usage
        return CachedLookup(embeddings=embeddings, context=init_context)

    def expand(self, t):
        """
        Expand tensor or dictionary of tensors for POMO.

        Args:
            t (torch.Tensor or dict or None): Input to expand.

        Returns:
            Expanded input.
        """
        if t is None:
            return None
        # Use ITensorDictLike protocol for dict-like tensor containers
        if isinstance(t, ITensorDictLike):
            return t.__class__({k: self.expand(v) for k, v in t.items()})  # type: ignore[call-arg]

        # Expand (Batch, ...) -> (Batch * POMO, ...)
        # We repeat the batch elements
        bs = t.size(0)
        shape = (bs, self.pomo_size) + t.shape[1:]
        return t.unsqueeze(1).expand(shape).reshape(-1, *t.shape[1:])
