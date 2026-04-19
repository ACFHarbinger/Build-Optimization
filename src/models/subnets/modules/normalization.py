"""
Normalization layers (Batch, Layer, Instance).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from logic.src.configs.models.normalization import NormalizationConfig


class Normalization(nn.Module):
    """
    Wrapper for various Normalization layers (Batch, Layer, Instance).
    """

    normalizer: nn.Module

    def __init__(
        self,
        embed_dim: int,
        norm_name: Optional[str] = None,
        eps_alpha: Optional[float] = None,
        learn_affine: Optional[bool] = None,
        track_stats: Optional[bool] = None,
        mbval: Optional[float] = None,
        n_groups: Optional[int] = None,
        kval: Optional[float] = None,
        bias: Optional[bool] = True,
        norm_config: Optional[NormalizationConfig] = None,
    ) -> None:
        """
        Initializes the normalization layer.

        Args:
            embed_dim: Embedding dimension.
            norm_name: Type of normalization ('batch', 'layer', 'instance', 'group', 'local_response').
            eps_alpha: Epsilon value for numerical stability.
            learn_affine: If True, learn affine parameters (weight and bias).
            track_stats: If True, track running statistics for BatchNorm/InstanceNorm.
            mbval: Momentum for running statistics or beta for LocalResponseNorm.
            n_groups: Number of groups for GroupNorm.
            kval: k value for LocalResponseNorm.
            bias: If True, add bias for LayerNorm.
            norm_config: Normalization configuration object.
        """
        super(Normalization, self).__init__()

        # Use norm_config if provided, otherwise create from individual args
        if norm_config is None:
            norm_config = NormalizationConfig(
                norm_type=norm_name if norm_name is not None else "batch",
                epsilon=eps_alpha if eps_alpha is not None else 1e-05,
                learn_affine=learn_affine if learn_affine is not None else True,
                track_stats=track_stats if track_stats is not None else False,
                momentum=mbval if mbval is not None else 0.1,
                n_groups=n_groups if n_groups is not None else 3,
                k_lrnorm=kval if kval is not None else 1.0,
            )

        norm_name = norm_config.norm_type
        eps_alpha = norm_config.epsilon
        learn_affine = norm_config.learn_affine
        track_stats = norm_config.track_stats
        mbval = norm_config.momentum
        n_groups = norm_config.n_groups
        kval = norm_config.k_lrnorm

        if norm_name == "instance":
            self.normalizer = nn.InstanceNorm1d(
                embed_dim,
                eps=eps_alpha,
                affine=learn_affine if learn_affine is not None else True,
                track_running_stats=track_stats if track_stats is not None else False,
                momentum=mbval if mbval is not None else 0.1,
            )
        elif norm_name == "batch":
            self.normalizer = nn.BatchNorm1d(
                embed_dim,
                eps=eps_alpha,
                affine=learn_affine if learn_affine is not None else True,
                track_running_stats=track_stats if track_stats is not None else False,
                momentum=mbval if mbval is not None else 0.1,
            )
        elif norm_name == "layer":
            self.normalizer = nn.LayerNorm(
                embed_dim,
                eps=eps_alpha,
                elementwise_affine=learn_affine if learn_affine is not None else True,
                bias=bias if bias is not None else True,
            )
        elif norm_name == "group":
            if n_groups is None or embed_dim % n_groups != 0:
                n_groups = 1
            self.normalizer = nn.GroupNorm(
                num_groups=n_groups,
                num_channels=embed_dim,
                eps=eps_alpha,
                affine=learn_affine if learn_affine is not None else True,
            )
        elif norm_name == "local_response":
            self.normalizer = nn.LocalResponseNorm(
                embed_dim,
                alpha=eps_alpha,
                beta=mbval if mbval is not None else 0.75,
                k=kval if kval is not None else 1.0,
            )
        else:
            raise ValueError(f"Unknown normalization method: {norm_name}")

        if learn_affine:
            self.init_parameters()

    def init_parameters(self) -> None:
        """Initializes the affine parameters if applicable."""
        for param in self.parameters():
            if param.dim() > 0:
                stdv: float = 1.0 / math.sqrt(param.size(-1))
                param.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies the normalization to the input.

        Args:
            input: Input tensor.
            mask: Optional mask (currently not used by normalization layers).

        Returns:
            Normalized tensor.
        """
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, (nn.InstanceNorm1d, nn.GroupNorm)):
            # Normalize over embedding dimension (last)
            # We need to move the last dimension to the C position (index 1)
            # and potentially flatten intermediate dimensions if normalizer is 1D
            orig_shape = input.shape
            dims = input.dim()
            if dims > 3:
                # view as (B, N1*N2*..., D)
                curr = input.view(orig_shape[0], -1, orig_shape[-1])
                # permute to (B, D, N_total)
                curr = curr.permute(0, 2, 1)
                curr = self.normalizer(curr)
                # permute back and reshape
                return curr.permute(0, 2, 1).view(*orig_shape)
            elif dims == 3:
                return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                return self.normalizer(input)
        elif isinstance(self.normalizer, nn.LayerNorm):
            return self.normalizer(input)
        else:
            return self.normalizer(input)
