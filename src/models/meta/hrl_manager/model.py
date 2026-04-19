"""
This module contains the GAT-LSTM Manager agent implementation for Hierarchical RL.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.constants.models import (
    DEFAULT_TEMPORAL_HORIZON,
    STATIC_DIM,
)
from logic.src.constants.routing import DEFAULT_EVAL_BATCH_SIZE
from logic.src.constants.waste import CRITICAL_FILL_THRESHOLD
from logic.src.models.subnets.factories import AttentionComponentFactory, NeuralComponentFactory
from logic.src.tracking.logging.pylogger import get_pylogger

from .critic_head import CriticHead
from .gate_head import GateHead
from .mandatory_head import MandatorySelectionHead
from .temporal_encoder import TemporalEncoder

logger = get_pylogger(__name__)


class MandatoryManager(nn.Module):
    """
    Hierarchical Manager Agent for PCVRP.
    Uses RNN to encode temporal waste patterns and GNN to encode spatial structure.
    Outputs:
        1. Mandatory Selection: Which bins must be collected today (mandatory=True means mandatory).
        2. Route Gate: Whether to dispatch vehicles today.
    """

    def __init__(
        self,
        input_dim_static=STATIC_DIM,  # x, y
        input_dim_dynamic=DEFAULT_TEMPORAL_HORIZON,  # waste history length
        global_input_dim=2,  # avg_waste, max_waste (Only current)
        critical_threshold=CRITICAL_FILL_THRESHOLD,
        batch_size=DEFAULT_EVAL_BATCH_SIZE,
        hidden_dim=128,
        embed_dim: Optional[int] = None,
        feed_forward_hidden: Optional[int] = None,
        lstm_hidden=64,
        num_layers_gat=3,
        n_heads=8,
        dropout=0.1,
        device="cuda",
        shared_encoder=None,
        temporal_encoder_cls=None,
        temporal_encoder_kwargs=None,
        spatial_encoder_cls=None,
        spatial_encoder_kwargs=None,
        component_factory: NeuralComponentFactory = None,  # type: ignore[assignment]
        temporal_encoder_type: str = "lstm",
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
    ):
        """
        Initialize the MandatoryManager.

        Args:
            input_dim_static (int, optional): Static input dimension (e.g., coordinates). Defaults to 2.
            input_dim_dynamic (int, optional): Dynamic input dimension (e.g., waste history). Defaults to 10.
            global_input_dim (int, optional): Global input dimension (e.g., avg waste). Defaults to 2.
            critical_threshold (float, optional): Threshold for critical waste levels. Defaults to 0.9.
            batch_size (int, optional): Batch size. Defaults to 1024.
            hidden_dim (int, optional): Hidden dimension size (legacy, use embed_dim/feed_forward_hidden). Defaults to 128.
            embed_dim (int, optional): Embedding dimension.
            feed_forward_hidden (int, optional): Feed-forward hidden dimension.
            lstm_hidden (int, optional): LSTM hidden dimension. Defaults to 64.
            num_layers_gat (int, optional): Number of GAT layers. Defaults to 3.
            n_heads (int, optional): Number of attention heads. Defaults to 8.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            device (str, optional): Computation device. Defaults to 'cuda'.
            shared_encoder (nn.Module, optional): Shared encoder instance. Defaults to None.
            temporal_encoder_cls (class, optional): Class for temporal encoder. Defaults to TemporalEncoder.
            temporal_encoder_kwargs (dict, optional): Kwargs for temporal encoder.
            spatial_encoder_cls (class, optional): Class for spatial encoder. Defaults to TransformerEncoder.
            spatial_encoder_kwargs (dict, optional): Kwargs for spatial encoder.
            component_factory (NeuralComponentFactory, optional): Factory for creating neural components.
            temporal_encoder_type (str, optional): Type of temporal encoder. Defaults to "lstm".
            norm_config (NormalizationConfig, optional): Configuration for normalization.
            activation_config (ActivationConfig, optional): Configuration for activation functions.
        """
        super(MandatoryManager, self).__init__()
        self.device = device
        self.batch_size = batch_size

        if norm_config is None:
            norm_config = NormalizationConfig()

        if activation_config is None:
            activation_config = ActivationConfig()

        self.embed_dim, self.ff_hidden = self._init_shared_encoder_dim(
            shared_encoder, embed_dim or hidden_dim, feed_forward_hidden or (embed_dim or hidden_dim) * 4
        )
        self.hidden_dim = self.embed_dim  # For backwards compatibility with internal refs
        self.input_dim_dynamic = input_dim_dynamic
        self.global_input_dim = global_input_dim
        self.critical_threshold = critical_threshold

        # 1. Temporal Encoder
        self.temporal_encoder = self._init_temporal_encoder(
            temporal_encoder_cls, temporal_encoder_kwargs, lstm_hidden, temporal_encoder_type
        )

        # 2. Feature Fusion
        # Static (2) + Temporal Embedding (lstm_hidden) -> Hidden
        # NOTE: We assume the temporal encoder output dim corresponds to lstm_hidden
        # If a custom encoder is used with different output dim, user must ensure consistency
        self.feature_embedding = nn.Linear(input_dim_static + lstm_hidden, hidden_dim)

        self.gat_encoder = self._init_spatial_encoder(
            shared_encoder=shared_encoder,
            component_factory=component_factory,
            spatial_encoder_cls=spatial_encoder_cls,
            spatial_encoder_kwargs=spatial_encoder_kwargs,
            embed_dim=self.embed_dim,
            feed_forward_hidden=self.ff_hidden,
            num_layers_gat=num_layers_gat,
            n_heads=n_heads,
            dropout=dropout,
            norm_config=norm_config,
            activation_config=activation_config,
        )

        # 4. Heads
        self._init_heads(self.embed_dim, lstm_hidden, global_input_dim)

        self.to(device)

        # Buffers for PPO
        self.clear_memory()

        # Force the Manager to prefer Routing (1) over Skipping (0) at the start
        self._initialize_head_weights()

    def _init_shared_encoder_dim(
        self, shared_encoder: Optional[nn.Module], embed_dim: int, hidden_dim: int
    ) -> Tuple[int, int]:
        """
        Initialize the dimensions based on shared encoder.

        Args:
            shared_encoder: Optional shared encoder module.
            embed_dim: Default embedding dimension.
            hidden_dim: Default hidden (feed-forward) dimension.

        Returns:
            Tuple[int, int]: (embed_dim, hidden_dim).
        """
        if shared_encoder is not None:
            # Prefer shared encoder attributes if they exist
            e_dim = getattr(shared_encoder, "embed_dim", getattr(shared_encoder, "hidden_dim", embed_dim))
            h_dim = getattr(shared_encoder, "feed_forward_hidden", getattr(shared_encoder, "hidden_dim", hidden_dim))
            return e_dim, h_dim
        return embed_dim, hidden_dim

    def _init_temporal_encoder(
        self,
        temporal_encoder_cls: Optional[type] = None,
        temporal_encoder_kwargs: Optional[dict] = None,
        lstm_hidden: int = 64,
        temporal_encoder_type: str = "lstm",
    ) -> nn.Module:
        """
        Initialize the temporal encoder.

        Args:
            temporal_encoder_cls: Optional class for temporal encoder.
            temporal_encoder_kwargs: Optional kwargs for temporal encoder.
            lstm_hidden: LSTM hidden dimension.
            temporal_encoder_type: Type of temporal encoder ('lstm', 'gru').

        Returns:
            nn.Module: Temporal encoder module.
        """
        if temporal_encoder_cls is not None:
            kwargs = temporal_encoder_kwargs or {}
            return temporal_encoder_cls(**kwargs)

        return TemporalEncoder(hidden_dim=lstm_hidden, rnn_type=temporal_encoder_type)

    def _init_spatial_encoder(
        self,
        shared_encoder: Optional[nn.Module] = None,
        component_factory: Optional[NeuralComponentFactory] = None,
        spatial_encoder_cls: Optional[type] = None,
        spatial_encoder_kwargs: Optional[dict] = None,
        embed_dim: int = 128,
        feed_forward_hidden: int = 512,
        num_layers_gat: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
    ) -> nn.Module:
        """
        Initialize the spatial encoder.

        Args:
            shared_encoder: Optional shared encoder instance.
            component_factory: Optional factory for creating neural components.
            spatial_encoder_cls: Optional class for spatial encoder.
            spatial_encoder_kwargs: Optional kwargs for spatial encoder.
            hidden_dim: Hidden dimension size.
            num_layers_gat: Number of GAT layers.
            n_heads: Number of attention heads.
            dropout: Dropout rate.
            norm_config: Configuration for normalization.
            activation_config: Configuration for activation functions.

        Returns:
            nn.Module: Spatial encoder module.
        """
        if shared_encoder is not None:
            return shared_encoder

        if component_factory is not None:
            return component_factory.create_encoder(
                norm_config=norm_config,
                activation_config=activation_config,
                embed_dim=embed_dim,
                feed_forward_hidden=feed_forward_hidden,
                n_layers=num_layers_gat,
                n_heads=n_heads,
                dropout_rate=dropout,
            )

        if spatial_encoder_cls is not None:
            kwargs = spatial_encoder_kwargs or {}
            return spatial_encoder_cls(**kwargs)

        factory = AttentionComponentFactory()
        return factory.create_encoder(
            norm_config=norm_config,
            activation_config=activation_config,
            embed_dim=embed_dim,
            feed_forward_hidden=feed_forward_hidden,
            n_layers=num_layers_gat,
            n_heads=n_heads,
            dropout_rate=dropout,
        )

    def _init_heads(self, hidden_dim: int, lstm_hidden: int, global_input_dim: int):
        """Initialize task-specific heads."""
        # Mandatory Selection Head (Actor 1)
        # Output: (Batch, N, 2) -> Logits for [Optional, Mandatory]
        # Index 0: bin is optional (can skip), Index 1: bin must be collected
        # Input: Hidden + Skip Connection (lstm_hidden)
        self.mandatory_head = MandatorySelectionHead(input_dim=hidden_dim + lstm_hidden, hidden_dim=hidden_dim)

        # Route Gate Head (Actor 2)
        # Global Pooling -> MLP -> (Batch, 2) -> Logits for [No Route, Route]
        # Input: Hidden + Global Features
        self.gate_head = GateHead(input_dim=hidden_dim + global_input_dim, hidden_dim=hidden_dim)

        # Critic (Value Function)
        # Global Pooling -> MLP -> (Batch, 1)
        # Input: Hidden + Global Features
        self.critic = CriticHead(input_dim=hidden_dim + global_input_dim, hidden_dim=hidden_dim)

    def _initialize_head_weights(self):
        """Force the Manager to prefer Routing (1) over Skipping (0) at the start."""
        with torch.no_grad():
            # Assuming gate_head's last layer is a Linear layer
            # Accessing net[-1] from GateHead wrapper
            self.gate_head.net[-1].bias.fill_(0)

            # Neutral bias for mandatory_head
            self.mandatory_head.net[-1].bias.fill_(0)

    def clear_memory(self):
        """
        Clear the memory buffers used for PPO.
        """
        self.states_static = []
        self.states_dynamic = []
        self.states_global = []
        self.actions_mandatory = []  # Which bins must be collected
        self.actions_gate = []
        self.log_probs_mandatory = []
        self.log_probs_gate = []
        self.rewards = []
        self.values = []
        self.target_mandatory = []  # Ground truth mandatory for auxiliary loss
        self.dones = []

    def feature_processing(self, static, dynamic):
        """
        Process static and dynamic features using LSTM and Linear layers.

        Args:
            static (torch.Tensor): Static features (Batch, N, 2).
            dynamic (torch.Tensor): Dynamic features (Batch, N, History).

        Returns:
            tuple: (x, temporal_embed)
                   - x: Combined features projected to hidden dimension (Batch, N, Hidden).
                   - temporal_embed: Temporal embeddings from LSTM (Batch, N, lstm_hidden).
        """
        temporal_embed = self.temporal_encoder(dynamic)

        # Concatenate with static
        # static: (Batch, N, 2)
        combined = torch.cat([static, temporal_embed], dim=2)

        # Project to hidden
        x = self.feature_embedding(combined)

        return x, temporal_embed

    def forward(self, static, dynamic, global_features):
        """
        Forward pass of the Manager Agent.

        Args:
            static (torch.Tensor): Static features (Batch, N, 2) - coordinates.
            dynamic (torch.Tensor): Dynamic features (Batch, N, History) - waste history.
            global_features (torch.Tensor): Global features (Batch, global_input_dim).

        Returns:
            tuple: (mandatory_logits, gate_logits, value)
                   - mandatory_logits: Logits for mandatory selection (B, N, 2).
                     Index 0 = optional, Index 1 = must collect.
                   - gate_logits: Logits for route gating (B, 2).
                     Index 0 = skip routing, Index 1 = dispatch vehicles.
                   - value: Estimated state value (B, 1).
        """
        # Encode
        x, temporal_embed = self.feature_processing(static, dynamic)  # (B, N, H)

        # Spatial Context
        # Transformer expects (Batch, Seq, F) with batch_first=True
        h = self.gat_encoder(x)  # (B, N, H)

        # SKIP CONNECTION: Concatenate temporal_embed (LSTM output) to h
        # h_skip: (B, N, H + lstm_hidden)
        h_skip = torch.cat([h, temporal_embed], dim=-1)

        # Mandatory Selection Logits
        # Outputs logits for [Optional, Mandatory] per node
        mandatory_logits = self.mandatory_head(h_skip)  # (B, N, 2)

        # Global Pooling for Gate/Critic
        # Combine Mean (for general load) and Max (for urgency/overflows)
        h_mean = torch.mean(h, dim=1)  # (B, H)
        h_max = torch.max(h, dim=1)[0]  # (B, H)
        h_global = h_mean + h_max  # Simple addition or concatenation

        # Concatenate Global Features
        # global_features: (B, G)
        h_combined = torch.cat([h_global, global_features], dim=1)  # (B, H+G)

        # Gate Logits
        gate_logits = self.gate_head(h_combined)  # (B, 2)

        # Value
        value = self.critic(h_combined)  # (B, 1)

        return mandatory_logits, gate_logits, value

    def select_action(
        self,
        static,
        dynamic,
        global_features=None,
        deterministic=False,
        threshold=0.5,
        mandatory_threshold=0.5,
        target_mandatory=None,
    ):
        """
        Select actions (gate and mandatory) based on the current state.

        Args:
            static (torch.Tensor): Static features (Batch, N, 2).
            dynamic (torch.Tensor): Dynamic features (Batch, N, History).
            global_features (torch.Tensor): Global features (Batch, G).
            deterministic (bool): If True, use argmax instead of sampling.
            threshold (float): Probability threshold for gate decision.
            mandatory_threshold (float): Probability threshold for mandatory decision.
            target_mandatory (torch.Tensor, optional): Ground truth mandatory for auxiliary loss.

        Returns:
            tuple: (mandatory_action, gate_action, value)
                   - mandatory_action: (Batch, N) boolean tensor where True = must collect.
                   - gate_action: (Batch,) long tensor where 1 = dispatch vehicles.
                   - value: (Batch, 1) state value estimate.
        """
        mandatory_logits, gate_logits, value = self.forward(static, dynamic, global_features)

        # 1. Gate Decision (0=Skip routing, 1=Dispatch vehicles)
        gate_probs = F.softmax(gate_logits, dim=-1)
        gate_dist = torch.distributions.Categorical(gate_probs)

        if deterministic:
            if threshold < 0:
                gate_action = torch.ones_like(gate_probs[:, 1]).long()
            else:
                gate_action = (gate_probs[:, 1] > threshold).long()
        else:
            gate_action = gate_dist.sample()

        # 2. Mandatory Selection (Which bins must be collected)
        # Index 0 = optional, Index 1 = must collect
        mandatory_probs = F.softmax(mandatory_logits, dim=-1)
        mandatory_dist = torch.distributions.Categorical(mandatory_probs)

        mandatory_action = (
            (mandatory_probs[:, :, 1] > mandatory_threshold).long() if deterministic else mandatory_dist.sample()
        )

        # Store for PPO
        if not deterministic:
            log_prob_mandatory = mandatory_dist.log_prob(mandatory_action).sum(dim=1)
            log_prob_gate = gate_dist.log_prob(gate_action)

            self.states_static.append(static.detach().cpu())
            self.states_dynamic.append(dynamic.detach().cpu())
            self.states_global.append(global_features.detach().cpu() if global_features is not None else None)
            self.actions_gate.append(gate_action.detach().cpu())
            self.actions_mandatory.append(mandatory_action.detach().cpu())
            self.log_probs_gate.append(log_prob_gate.detach().cpu())
            self.log_probs_mandatory.append(log_prob_mandatory.detach().cpu())
            self.values.append(value.detach().cpu())

            # Store target mandatory for auxiliary loss
            if target_mandatory is not None:
                self.target_mandatory.append(target_mandatory.detach().cpu())
            else:
                # Fallback: bins with fill > critical_threshold should be collected
                self.target_mandatory.append((dynamic[:, :, -1] > self.critical_threshold).float().detach().cpu())

        return mandatory_action, gate_action, value

    def get_mandatory_mask(
        self,
        static,
        dynamic,
        global_features=None,
        threshold=0.5,
    ):
        """
        Get the mandatory boolean mask for environment integration.

        This method provides the mandatory mask in the format expected by
        the WCVRP/VRPP environments for action masking.

        Args:
            static (torch.Tensor): Static features (Batch, N, 2).
            dynamic (torch.Tensor): Dynamic features (Batch, N, History).
            global_features (torch.Tensor): Global features (Batch, G).
            threshold (float): Probability threshold for mandatory decision.

        Returns:
            torch.Tensor: Boolean tensor (Batch, N) where True = must collect.
        """
        mandatory_logits, _, _ = self.forward(static, dynamic, global_features)
        mandatory_probs = F.softmax(mandatory_logits, dim=-1)

        # mandatory is True where P(mandatory=1) > threshold
        mandatory = mandatory_probs[:, :, 1] > threshold

        # Depot (index 0) should never be a mandatory target
        mandatory[:, 0] = False

        return mandatory
