"""
Temporal Attention Model for multi-day waste collection routing.

Extends the base AttentionModel with a GRU-based fill level predictor that uses
historical waste data to anticipate future bin fill levels. This enables proactive
collection decisions in stochastic waste scenarios (SCWCVRP, SWCVRP).

Architecture:
    Base AttentionModel + FillPredictor -> TemporalEmbedding -> CombineLayer

The temporal features are fused with static node embeddings before encoding,
allowing the attention mechanism to consider predicted future fill levels
when constructing routes.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.core.attention_model import AttentionModel
from logic.src.models.subnets.factories import NeuralComponentFactory
from logic.src.models.subnets.helpers.gru_fill_predictor import GatedRecurrentUnitFillPredictor
from logic.src.models.subnets.helpers.lstm_fill_predictor import LongShortTermMemoryFillPredictor
from logic.src.models.subnets.modules import ActivationFunction


class TemporalAttentionModel(AttentionModel):
    """
    Attention Model extended with temporal features (waste history).

    Integrates a fill level predictor and processes historical waste data.
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
        aggregation_graph: str = "mean",
        tanh_clipping: float = 10.0,
        mask_inner: bool = True,
        mask_logits: bool = True,
        mask_graph: bool = False,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        n_heads: int = 8,
        checkpoint_encoder: bool = False,
        shrink_size: Optional[int] = None,
        temporal_horizon: int = 5,
        predictor_layers: int = 2,
        pomo_size: int = 0,
        spatial_bias: bool = False,
        spatial_bias_scale: float = 1.0,
        entropy_weight: float = 0.0,
        connection_type: str = "residual",
        hyper_expansion: int = 4,
        decoder_type: str = "attention",
        predictor_type: str = "gru",
        **kwargs,
    ) -> None:
        """
        Initialize the Temporal Attention Model.
        """
        super().__init__(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            problem=problem,
            component_factory=component_factory,
            n_encode_layers=n_encode_layers,
            n_encode_sublayers=n_encode_sublayers,
            n_decode_layers=n_decode_layers,
            dropout_rate=dropout_rate,
            aggregation=aggregation,
            aggregation_graph=aggregation_graph,
            tanh_clipping=tanh_clipping,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            mask_graph=mask_graph,
            norm_config=norm_config,
            activation_config=activation_config,
            n_heads=n_heads,
            checkpoint_encoder=checkpoint_encoder,
            shrink_size=shrink_size,
            pomo_size=pomo_size,
            temporal_horizon=0,  # Explicitly set to 0 for initial context
            spatial_bias=spatial_bias,
            spatial_bias_scale=spatial_bias_scale,
            entropy_weight=entropy_weight,
            predictor_layers=predictor_layers,
            connection_type=connection_type,
            hyper_expansion=hyper_expansion,
            decoder_type=decoder_type,
            **kwargs,
        )

        if activation_config is None:
            activation_config = ActivationConfig()

        self.temporal_horizon = temporal_horizon
        if predictor_type == "lstm":
            self.fill_predictor = LongShortTermMemoryFillPredictor(
                input_dim=1,
                hidden_dim=hidden_dim,
                num_layers=predictor_layers,
                dropout=dropout_rate,
            )
        else:
            assert predictor_type == "gru", f"Unknown predictor type: {predictor_type}"
            self.fill_predictor = GatedRecurrentUnitFillPredictor(  # type: ignore[assignment]
                input_dim=1,
                hidden_dim=hidden_dim,
                num_layers=predictor_layers,
                dropout=dropout_rate,
            )

        self.temporal_embed = nn.Linear(1, embed_dim)
        self.predict_future = self.is_wc or self.is_vrpp

        self.combine_embeddings = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            ActivationFunction(
                activation_config.name,
                activation_config.param,
                activation_config.threshold,
                activation_config.replacement_value,
                activation_config.n_params,
                activation_config.range,  # type: ignore[arg-type]
            ),
            nn.Linear(embed_dim, embed_dim),
        )

    def _get_initial_embeddings(self, input: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[Any]]:
        """
        Get initial embeddings for nodes, incorporating predicted future fill levels.

        Args:
            input (dict): Input data containing node features and history.

        Returns:
            torch.Tensor: Combined embeddings (static + temporal).
        """
        # Get the base embeddings from context embedder (without temporal features)
        base_embeddings = self.context_embedder.init_node_embeddings(input)

        if "fill_history" not in list(input.keys()) or not self.predict_future:
            return base_embeddings, None

        fill_history = input["fill_history"]
        batch_size, graph_size, _ = fill_history.size()
        fill_history = fill_history.view(batch_size * graph_size, self.temporal_horizon, 1)

        predicted_fills = self.fill_predictor(fill_history)
        predicted_fills = predicted_fills.view(batch_size, graph_size, 1)

        # For depot node, set predicted fill to 0
        if self.is_vrpp or self.is_wc:
            depot_fill = torch.zeros((batch_size, 1, 1), device=predicted_fills.device)
            predicted_fills = torch.cat((depot_fill, predicted_fills), dim=1)

        fill_embeddings = self.temporal_embed(predicted_fills)
        combined_embeddings = self.combine_embeddings(torch.cat((base_embeddings, fill_embeddings), dim=-1))
        return combined_embeddings, None

    def forward(  # type: ignore[override]
        self,
        input,
        cost_weights=None,
        return_pi=False,
        pad=False,
        mask=None,
        expert_pi=None,
        **kwargs,
    ):
        """
        Forward pass of the Temporal Attention Model.

        Handles temporal feature prediction and updates if needed.

        Args:
            input (dict): The input data dictionary.
            cost_weights (torch.Tensor, optional): Weights for different cost components. Defaults to None.
            return_pi (bool, optional): Whether to return the action sequence. Defaults to False.
            pad (bool, optional): Whether to pad the solution sequence. Defaults to False.
            mask (torch.Tensor, optional): Mask for valid actions. Defaults to None.
            expert_pi (torch.Tensor, optional): Expert policy. Defaults to None.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple: (cost, log_likelihood, cost_dict, pi, entropy)
        """
        if "fill_history" not in list(input.keys()) and self.predict_future:
            locs_key = "locs" if "locs" in input else "loc"
            batch_size = input[locs_key].size(0)
            graph_size = input[locs_key].size(1)

            # For VRP-like problems, adjust for depot (excluded from graph size)
            if self.is_vrpp or self.is_wc:
                pass  # graph_size is already correct (num customers)

            fill_history = torch.zeros(
                (batch_size, graph_size, self.temporal_horizon),
                device=input[locs_key].device,
            )
            input["fill_history"] = fill_history
        return super().forward(
            input,
            env=cost_weights,
            return_pi=return_pi,
            pad=pad,
            mask=mask,
            expert_pi=expert_pi,
            **kwargs,
        )

    def update_fill_history(self, fill_history, new_fills):
        """
        Update the fill history with new fill levels (rolling window).

        Args:
            fill_history (torch.Tensor): Current fill history.
            new_fills (torch.Tensor): New fill levels to append.

        Returns:
            torch.Tensor: Updated fill history.
        """
        updated_history = fill_history.clone()
        updated_history[:, :, :-1] = fill_history[:, :, 1:]
        updated_history[:, :, -1] = new_fills
        return updated_history

    def compute_simulator_day(self, input, graph):
        """
        Compute one simulation day, updating fill history if present.

        Args:
            input (dict): Input data.
            graph (object): The graph object.

        Returns:
            dict: Simulation results.
        """
        if "fill_history" in list(input.keys()) and "current_fill" in list(input.keys()):
            input["fill_history"] = self.update_fill_history(input["fill_history"], input["current_fill"])
        return super().compute_simulator_day(input, graph)  # type: ignore[misc]
