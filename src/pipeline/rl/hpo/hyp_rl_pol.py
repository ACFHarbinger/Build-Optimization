from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from logic.src.models.subnets.modules import ActivationFunction, Normalization

from .base import ParamSpec
from .hyp_rl_enc import HyperparameterEncoder


class HypRLPolicy(nn.Module):
    """
    LSTM-based policy network for Hyp-RL.

    Processes the history of evaluated configurations and their rewards,
    then generates the next configuration to evaluate.
    """

    def __init__(
        self,
        search_space: Dict[str, ParamSpec],
        state_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
        normalization: str = "layer",
        activation: str = "relu",
        device: Optional[torch.device] = None,
    ):
        """Initialize the Hyp-RL policy network.

        Args:
            search_space: Dictionary mapping parameter names to specs.
            state_dim: Dimension of state embeddings.
            hidden_dim: Hidden dimension of LSTM.
            n_layers: Number of LSTM layers.
            normalization: Type of normalization.
            activation: Activation function name.
            device: Device to place tensors on.
        """
        super().__init__()
        self.search_space = search_space
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.device = device or torch.device("cpu")

        # Encoder for hyperparameter configurations
        self.config_encoder = HyperparameterEncoder(
            search_space,
            embed_dim=state_dim // 2,
            normalization=normalization,
            activation=activation,
            device=self.device,
        )

        # State consists of: config embedding + reward
        self.lstm = nn.LSTM(
            input_size=state_dim // 2 + 1,  # config embedding + reward
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        # Heads for generating next configuration
        self.param_heads = nn.ModuleDict()

        for name, spec in search_space.items():
            ptype = spec["type"]
            if ptype == "float" or ptype == "int":
                # Output mean and log_std for continuous distributions
                self.param_heads[name] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    Normalization(hidden_dim // 2, normalization, learn_affine=True),
                    ActivationFunction(activation),
                    nn.Linear(hidden_dim // 2, 2),  # [mean, log_std]
                )
            elif ptype == "categorical":
                # Output logits for categorical distribution
                n_choices = len(spec["choices"])
                self.param_heads[name] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    Normalization(hidden_dim // 2, normalization, learn_affine=True),
                    ActivationFunction(activation),
                    nn.Linear(hidden_dim // 2, n_choices),
                )

    def forward(
        self,
        history: List[Tuple[Dict[str, Any], float]],
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Generate next hyperparameter configuration.

        Args:
            history: List of (config, reward) tuples representing evaluation history.
            hidden: Optional LSTM hidden state from previous step.

        Returns:
            Tuple of:
                - sampled_config: Dictionary of sampled hyperparameter values
                - log_probs: Dictionary of log probabilities for each parameter
                - new_hidden: Updated LSTM hidden state
        """
        if len(history) == 0:
            # Initialize with zeros if no history
            batch_size = 1
            seq_len = 1
            lstm_input = torch.zeros(batch_size, seq_len, self.state_dim // 2 + 1, device=self.device)
        else:
            # Encode history
            embeddings = []
            for config, reward in history:
                config_emb = self.config_encoder(config)
                reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
                state = torch.cat([config_emb, reward_tensor], dim=-1)
                embeddings.append(state)

            lstm_input = torch.stack(embeddings, dim=0).unsqueeze(0).to(self.device)  # [1, seq_len, state_dim]

        # Process through LSTM
        if hidden is None:
            lstm_out, new_hidden = self.lstm(lstm_input)
        else:
            lstm_out, new_hidden = self.lstm(lstm_input, hidden)

        # Use last output
        context = lstm_out[:, -1, :]  # [1, hidden_dim]

        # Generate parameters
        sampled_config = {}
        log_probs = {}

        for name, spec in self.search_space.items():
            ptype = spec["type"]
            head_output = self.param_heads[name](context).squeeze(0)  # [output_dim]

            if ptype == "float":
                # Sample from normal distribution
                mean, log_std = head_output[0], head_output[1]
                std = torch.exp(log_std).clamp(min=1e-6)
                dist = Normal(mean, std)

                # Sample and clip to valid range
                normalized_value = torch.tanh(dist.rsample())  # [-1, 1]
                normalized_value = (normalized_value + 1) / 2  # [0, 1]

                low, high = spec["low"], spec["high"]
                if spec.get("log", False):
                    # Log scale
                    log_low, log_high = np.log(low), np.log(high)
                    log_value = log_low + normalized_value.item() * (log_high - log_low)
                    value = np.exp(log_value)
                else:
                    value = low + normalized_value.item() * (high - low)

                # Apply step if specified
                if "step" in spec:
                    value = np.round(value / spec["step"]) * spec["step"]

                sampled_config[name] = float(value)
                log_probs[name] = dist.log_prob(torch.atanh(normalized_value * 2 - 1))

            elif ptype == "int":
                # Sample from normal distribution and round
                mean, log_std = head_output[0], head_output[1]
                std = torch.exp(log_std).clamp(min=1e-6)
                dist = Normal(mean, std)

                normalized_value = torch.tanh(dist.rsample())
                normalized_value = (normalized_value + 1) / 2

                low, high = spec["low"], spec["high"]
                value = int(low + normalized_value.item() * (high - low))

                # Apply step if specified
                step = spec.get("step", 1)
                value = int(np.round((value - low) / step) * step + low)
                value = np.clip(value, low, high)

                sampled_config[name] = int(value)
                log_probs[name] = dist.log_prob(torch.atanh(normalized_value * 2 - 1))

            elif ptype == "categorical":
                # Sample from categorical distribution
                logits = head_output
                cat_dist = Categorical(logits=logits)
                idx = cat_dist.sample()
                value = spec["choices"][idx.item()]

                sampled_config[name] = value
                log_probs[name] = cat_dist.log_prob(idx)

        return sampled_config, log_probs, new_hidden
