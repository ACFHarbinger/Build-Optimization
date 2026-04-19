from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from logic.src.models.subnets.modules import ActivationFunction, Normalization

from .base import ParamSpec


class HyperparameterEncoder(nn.Module):
    """
    Encodes hyperparameter configurations into fixed-size embeddings.

    Handles mixed types (continuous, discrete, categorical) by creating
    appropriate embeddings for each parameter type.
    """

    def __init__(
        self,
        search_space: Dict[str, ParamSpec],
        embed_dim: int = 32,
        normalization: str = "layer",
        activation: str = "relu",
        device: Optional[torch.device] = None,
    ):
        """Initialize the hyperparameter encoder.

        Args:
            search_space: Dictionary mapping parameter names to specs.
            embed_dim: Dimension of output embeddings.
            normalization: Type of normalization to use.
            activation: Activation function name.
            device: Device to place tensors on.
        """
        super().__init__()
        self.search_space = search_space
        self.embed_dim = embed_dim
        self.device = device or torch.device("cpu")

        # Build encoding layers for each parameter
        self.encoders = nn.ModuleDict()
        self.param_names = list(search_space.keys())
        self.choice_to_idx_maps: Dict[str, Dict[Any, int]] = {}

        for name, spec in search_space.items():
            ptype = spec["type"]
            if ptype == "float" or ptype == "int":
                # Continuous/discrete parameters: linear projection
                self.encoders[name] = nn.Sequential(
                    nn.Linear(1, embed_dim),
                    Normalization(embed_dim, normalization, learn_affine=True),
                    ActivationFunction(activation),
                )
            elif ptype == "categorical":
                # Categorical parameters: embedding layer
                n_choices = len(spec["choices"])
                encoding_layer = nn.Embedding(n_choices, embed_dim)
                self.encoders[name] = encoding_layer
                # Store mapping from value to index
                self.choice_to_idx_maps[name] = {choice: idx for idx, choice in enumerate(spec["choices"])}

        # Combine all parameter embeddings
        total_dim = len(self.param_names) * embed_dim
        self.combiner = nn.Sequential(
            nn.Linear(total_dim, embed_dim * 2),
            Normalization(embed_dim * 2, normalization, learn_affine=True),
            ActivationFunction(activation),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, config: Dict[str, Any]) -> torch.Tensor:
        """Encode a hyperparameter configuration.

        Args:
            config: Dictionary mapping parameter names to values.

        Returns:
            Tensor of shape [embed_dim] representing the configuration.
        """
        embeddings = []

        for name in self.param_names:
            value = config[name]
            spec = self.search_space[name]
            ptype = spec["type"]

            if ptype == "float":
                # Normalize to [0, 1] range
                low, high = spec["low"], spec["high"]
                if spec.get("log", False):
                    # Log scale normalization
                    value = np.log(value)
                    low, high = np.log(low), np.log(high)
                normalized = (value - low) / (high - low + 1e-8)
                tensor = torch.tensor([[normalized]], dtype=torch.float32, device=self.device)
                emb = self.encoders[name](tensor).squeeze(0)
            elif ptype == "int":
                # Normalize to [0, 1] range
                low, high = spec["low"], spec["high"]
                normalized = (value - low) / (high - low + 1e-8)
                tensor = torch.tensor([[normalized]], dtype=torch.float32, device=self.device)
                emb = self.encoders[name](tensor).squeeze(0)
            elif ptype == "categorical":
                # Look up embedding using structured map
                choice_to_idx = self.choice_to_idx_maps[name]
                idx = choice_to_idx[value]
                tensor = torch.tensor([idx], dtype=torch.long, device=self.device)
                emb = self.encoders[name](tensor).squeeze(0)
            else:
                raise ValueError(f"Unknown parameter type: {ptype}")

            embeddings.append(emb)

        # Concatenate and combine
        combined = torch.cat(embeddings, dim=-1)
        return self.combiner(combined)
