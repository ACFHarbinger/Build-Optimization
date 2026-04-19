"""decoder.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import decoder
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
from torch import nn

from logic.src.models.subnets.decoders.glimpse.decoder import GlimpseDecoder


class MatNetDecoder(GlimpseDecoder):
    """
    Decoder for MatNet.
    Extends GlimpseDecoder to handle row and column embeddings from MatNetEncoder.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        problem: Any,
        n_heads: int = 8,
        tanh_clipping: float = 10.0,
        **kwargs,
    ):
        """Initialize Class.

        Args:
            embed_dim (int): Description of embed_dim.
            hidden_dim (int): Description of hidden_dim.
            problem (Any): Description of problem.
            n_heads (int): Description of n_heads.
            tanh_clipping (float): Description of tanh_clipping.
            kwargs (Any): Description of kwargs.
        """
        super().__init__(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            problem=problem,
            n_heads=n_heads,
            tanh_clipping=tanh_clipping,
            **kwargs,
        )
        self.project_col_context = nn.Linear(embed_dim, embed_dim, bias=False)
        self.strategy = "greedy"
        self.temp = 1.0

    @property
    def device(self) -> torch.device:
        """Get device of the model."""
        return next(self.parameters()).device

    def _precompute(self, embeddings: torch.Tensor, num_steps: int = 1) -> Any:
        """precompute.

        Args:
            embeddings (torch.Tensor): Description of embeddings.
            num_steps (int): Description of num_steps.

        Returns:
            Any: Description of return value.
        """
        # Standard precompute logic
        fixed = super()._precompute(embeddings, num_steps)
        return fixed

    def forward(  # type: ignore[override]
        self,
        input: Union[torch.Tensor, dict[str, torch.Tensor]],
        embeddings: torch.Tensor,
        cost_weights: Optional[torch.Tensor] = None,
        dist_matrix: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        expert_pi: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Matrix-aware forward pass.
        Expects 'col_embeddings' in kwargs.
        """
        col_embeddings = kwargs.get("col_embeddings")
        if col_embeddings is None:
            col_embeddings = embeddings

        state = self.problem.make_state(input, cost_weights=cost_weights)
        fixed = super()._precompute(embeddings)

        # Add column information to fixed context
        col_avg = col_embeddings.mean(1)
        fixed.context_node_projected = fixed.context_node_projected + self.project_col_context(col_avg)[:, None, :]  # type: ignore[misc]

        outputs = []
        sequences = []

        while not state.all_finished().all():
            get_log_p_out = self._get_log_p(fixed, state)
            if not isinstance(get_log_p_out, tuple) or len(get_log_p_out) != 2:
                # This should not happen unless _get_log_p is mocked or broken
                log_p = get_log_p_out
                mask_out = None
            else:
                log_p, mask_out = get_log_p_out

            # Use mask_out if provided, otherwise generic mask
            current_mask = mask_out if mask_out is not None else mask

            # Select node
            probs = log_p.exp()[:, 0, :]  # type: ignore[union-attr]
            m = current_mask[:, 0, :] if current_mask is not None else None
            selected = self._select_node(probs, m)

            state = state.update(selected)
            outputs.append(log_p[:, 0, :])  # type: ignore[call-overload]
            sequences.append(selected)

        if not outputs:
            return torch.zeros(state.ids.size(0), 0, embeddings.size(-1), device=embeddings.device), torch.zeros(
                state.ids.size(0), 0, dtype=torch.long, device=embeddings.device
            )

        return torch.stack(outputs, 1), torch.stack(sequences, 1)
