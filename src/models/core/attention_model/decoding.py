"""
Decoding logic for AttentionModel.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


class DecodingMixin:
    """Mixin for decoding strategies (Greedy, Sampling, Beam Search, POMO)."""

    def __init__(self) -> None:
        """Initialize Class.

        Args:
            None.
        """
        # Type hints
        self.strategy: str = "greedy"
        self.temp: Optional[float] = None
        self.problem: Any

    def set_strategy(self, strategy: str, temp: Optional[float] = None):
        """
        Set the decoding strategy for the model.

        Args:
            strategy: The decoding strategy ('greedy' or 'sampling').
            temp: Temperature for sampling. Defaults to None.
        """
        self.strategy = strategy
        if temp is not None:
            self.temp = temp

    def beam_search(self, *args: Any, **kwargs: Any):
        """
        Perform beam search decoding.

        Args:
            *args: Variable length argument list passed to the problem's beam_search.
            **kwargs: Arbitrary keyword arguments passed to the problem's beam_search.

        Returns:
            The result of the beam search.
        """
        return self.problem.beam_search(*args, **kwargs, model=self)

    def propose_expansions(
        self,
        beam: Any,
        fixed: Any,
        expand_size: Optional[int] = None,
        normalize: bool = False,
        max_calc_batch_size: int = 4096,
    ):
        """
        Propose expansions for beam search.

        Args:
            beam: The current beam state.
            fixed: The precomputed fixed embeddings.
            expand_size: The number of expansions to propose. Defaults to None.
            normalize: Whether to normalize probabilities. Defaults to False.
            max_calc_batch_size: Max batch size for calculation. Defaults to 4096.

        Returns:
            Proposed expansions (indices and probabilities).
        """
        # Delegate to decoder's detailed proposal method if available
        # logic.src.models.subnets.modules/decoder.py usually handles this
        return self.decoder.propose_expansions(beam, fixed, expand_size, normalize, max_calc_batch_size)  # type: ignore[attr-defined]

    def sample_many(
        self,
        input: Dict[str, Any],
        cost_weights: Optional[torch.Tensor] = None,
        batch_rep: int = 1,
        iter_rep: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample multiple solutions for the same input (e.g., for POMO or validation).

        Args:
            input: The input data.
            cost_weights: Weights for different cost components. Defaults to None.
            batch_rep: Batch replication factor (for POMO).
            iter_rep: Iteration replication factor (for multiple sampling passes).

        Returns:
            tuple: (costs, pis)
                costs: Tensor of costs for each sampled solution.
                pis: Tensor of action indices for each sampled solution.
        """
        from logic.src.utils.functions import sample_many as _sample_many

        return _sample_many(
            lambda i: self.forward(  # type: ignore[attr-defined]
                i,
                strategy="sampling",
                return_pi=True,
                cost_weights=cost_weights,
            ),
            input,  # type: ignore[arg-type]
            batch_rep,
            iter_rep,
        )
