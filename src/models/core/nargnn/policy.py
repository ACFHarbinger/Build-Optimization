"""
NARGNN Policy.

Non-autoregressive GNN-based policy for combinatorial optimization.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from tensordict import TensorDict
from torch import nn

from logic.src.envs import get_env
from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.non_autoregressive.decoder import NonAutoregressiveDecoder
from logic.src.models.common.non_autoregressive.encoder import NonAutoregressiveEncoder
from logic.src.models.common.non_autoregressive.policy import NonAutoregressivePolicy
from logic.src.models.subnets.decoders.nar import SimpleNARDecoder
from logic.src.models.subnets.encoders.nargnn import NARGNNEncoder


class NARGNNPolicy(NonAutoregressivePolicy):
    """
    Base Non-autoregressive policy for NCO construction methods.

    Creates a heatmap of NxN for N nodes (i.e., heuristic) that models
    the probability to go from one node to another for all nodes.

    The policy performs the following steps:
        1. Encode the environment initial state into node embeddings
        2. Decode (non-autoregressively) to construct the solution

    Warning:
        The effectiveness of the non-autoregressive approach can vary
        significantly across different problem types and configurations.
        It may require careful tuning of the model architecture and
        decoding strategy to achieve competitive results.

    Args:
        encoder: Encoder module. Can be passed by sub-classes.
        decoder: Decoder module. Defaults to non-autoregressive decoder.
        embed_dim: Dimension of the embeddings.
        env_name: Name of the environment used to initialize embeddings.
        init_embedding: Model to use for the initial embedding. If None, use default.
        edge_embedding: Model to use for the edge embedding. If None, use default.
        graph_network: Model to use for the graph network. If None, use default.
        heatmap_generator: Model to use for the heatmap generator. If None, use default.
        num_layers_heatmap_generator: Number of layers in the heatmap generator.
        num_layers_graph_encoder: Number of layers in the graph encoder.
        act_fn: Activation function to use in the encoder.
        agg_fn: Aggregation function to use in the encoder.
        linear_bias: Whether to use bias in the encoder.
        train_strategy: Type of decoding during training.
        val_strategy: Type of decoding during validation.
        test_strategy: Type of decoding during testing.
        **constructive_policy_kw: Additional keyword arguments.
    """

    def __init__(
        self,
        encoder: Optional[NonAutoregressiveEncoder] = None,
        decoder: Optional[NonAutoregressiveDecoder] = None,
        embed_dim: int = 64,
        env_name: str = "tsp",
        init_embedding: Optional[nn.Module] = None,
        edge_embedding: Optional[nn.Module] = None,
        graph_network: Optional[nn.Module] = None,
        heatmap_generator: Optional[nn.Module] = None,
        num_layers_heatmap_generator: int = 5,
        num_layers_graph_encoder: int = 15,
        act_fn: str = "silu",
        agg_fn: str = "mean",
        linear_bias: bool = True,
        train_strategy: str = "sampling",
        val_strategy: str = "greedy",
        test_strategy: str = "greedy",
        **constructive_policy_kw,
    ) -> None:
        """
        Initialize NARGNNPolicy.

        Args:
           encoder: Encoder instance (optional).
           decoder: Decoder instance (optional).
           embed_dim: Embedding dimension.
           env_name: Environment name.
           init_embedding: Initial embedding module.
           edge_embedding: Edge embedding module.
           graph_network: Graph network module.
           heatmap_generator: Heatmap generator module.
           num_layers_heatmap_generator: Layers in heatmap generator.
           num_layers_graph_encoder: Layers in graph encoder.
           act_fn: Activation function.
           agg_fn: Aggregation function.
           linear_bias: Use bias in linear layers.
           train_strategy: Strategy for training.
           val_strategy: Strategy for validation.
           test_strategy: Strategy for testing.
           **constructive_policy_kw: Args for NonAutoregressivePolicy.
        """
        if encoder is None:
            # NARGNNEncoder doesn't inherit from NonAutoregressiveEncoder but has compatible interface
            encoder = NARGNNEncoder(  # type: ignore[assignment]
                embed_dim=embed_dim,
                env_name=env_name,
                init_embedding=init_embedding,
                edge_embedding=edge_embedding,
                graph_network=graph_network,
                heatmap_generator=heatmap_generator,
                num_layers_heatmap_generator=num_layers_heatmap_generator,
                num_layers_graph_encoder=num_layers_graph_encoder,
                act_fn=act_fn,
                agg_fn=agg_fn,
                linear_bias=linear_bias,
            )

        if decoder is None:
            decoder = SimpleNARDecoder()

        # Pass to constructive policy
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            **constructive_policy_kw,
        )

        self.train_strategy = train_strategy
        self.val_strategy = val_strategy
        self.test_strategy = test_strategy

    def forward(  # type: ignore[override]
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        num_starts: int = 1,
        phase: str = "test",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass: encode heatmap + decode solution.
        """
        # Encode: predict heatmap
        heatmap, node_embed = self.encoder(td)  # type: ignore[misc]

        # Strategy selection
        strategy = kwargs.get("strategy")
        if strategy is None:
            strategy = getattr(self, f"{phase}_strategy", "greedy")

        # Instantiate environment if needed
        if env is None:
            env = get_env(self.env_name)  # type: ignore[arg-type]

        # Use common_decoding
        logprobs, actions, td, env = self.common_decoding(
            strategy=strategy,  # type: ignore[arg-type]
            td=td,
            env=env,
            heatmap=heatmap,
            num_starts=num_starts,
            **kwargs,
        )

        # Constructed outputs
        # Narrow env type to RL4COEnvBase (guaranteed by common_decoding)
        # Post-decoding result preparation
        from logic.src.utils.decoding import get_log_likelihood

        out = {
            "actions": actions,
            "reward": env.get_reward(td, actions) if env is not None else None,
            "log_likelihood": get_log_likelihood(logprobs, None, td.get("mask", None), True),
            "heatmap": heatmap,
            "node_embed": node_embed,
        }
        return out
