"""
Base module for vectorized selection strategies.
"""

import functools
from abc import ABC, abstractmethod

from torch import Tensor

from logic.src.tracking.viz_mixin import PolicyVizMixin


class VectorizedSelector(PolicyVizMixin, ABC):
    """Abstract base class for vectorized bin selection strategies."""

    def __init_subclass__(cls, **kwargs) -> None:
        """Auto-wrap ``select`` in every concrete subclass to record telemetry."""
        super().__init_subclass__(**kwargs)
        if "select" in cls.__dict__:
            original = cls.__dict__["select"]

            @functools.wraps(original)
            def _instrumented(self, fill_levels: Tensor, **kw) -> Tensor:
                mask = original(self, fill_levels, **kw)
                day = kw.get("current_day")
                self._viz_record(
                    n_selected=int(mask[:, 1:].sum().item()),
                    mean_fill=float(fill_levels[:, 1:].mean().item()),
                    day=int(day.max().item()) if day is not None else -1,
                )
                return mask

            cls.select = _instrumented  # type: ignore[method-assign]

    @abstractmethod
    def select(
        self,
        fill_levels: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Select bins that must be collected.

        Args:
            fill_levels: Current fill levels (batch_size, num_nodes).
                         Values in [0, 1] where 1.0 = 100% full.
            **kwargs: Strategy-specific parameters.

        Returns:
            Tensor: Boolean mask (batch_size, num_nodes) where True = must collect.
                    Note: Index 0 is the depot and should always be False.
        """
        pass
