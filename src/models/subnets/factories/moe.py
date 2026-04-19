"""MoE component factory."""

from __future__ import annotations

from typing import Any

from torch import nn

from logic.src.models.subnets.encoders.moe import MoEGraphAttentionEncoder

from .base import NeuralComponentFactory, _create_decoder_by_type


class MoEComponentFactory(NeuralComponentFactory):
    """Factory for Mixture of Experts Models."""

    def __init__(self, num_experts: int = 4, k: int = 2, noisy_gating: bool = True) -> None:
        """
        Initialize the MoE Component Factory.

        Args:
            num_experts: Number of experts in the mixture.
            k: Number of experts to select per token.
            noisy_gating: Whether to add noise to the gating mechanism.
        """
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating

    def create_encoder(self, **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create MoE Graph Attention Encoder."""

        # Inject MoE params
        kwargs["num_experts"] = self.num_experts
        kwargs["k"] = self.k
        kwargs["noisy_gating"] = self.noisy_gating
        return MoEGraphAttentionEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "attention", **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create decoder based on decoder_type."""
        return _create_decoder_by_type(decoder_type, **kwargs)
