"""
Abstract generator for build-optimisation problem instances.

Mirrors WSmart-Route's Generator base class but generates item pools
rather than geographic node sets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Union

import torch
from tensordict import TensorDict


class BuildGenerator(ABC):
    """
    Abstract base class for synthetic build-optimisation instance generators.

    A generator creates a batched TensorDict containing a randomised item pool
    for one game type.  Each call returns ``batch_size`` independent instances.

    Subclasses must implement :meth:`_generate` and set the class-level
    attributes ``stat_names`` and ``num_slots``.

    Tensor shapes produced (B = batch dimension):

    ============== ==================== =====================================
    Key            Shape                Description
    ============== ==================== =====================================
    items          [B, N, S]            Normalised item stat features
    costs          [B, N]               Item acquisition cost
    slot_ids       [B, N]               Equipment slot index (0 … num_slots-1)
    rarities       [B, N]               Rarity multiplier (1.0 … 2.0)
    budget         [B]                  Total available budget
    ============== ==================== =====================================

    where N = num_slots × items_per_slot and S = len(stat_names).
    """

    stat_names: List[str] = []
    num_slots: int = 0

    def __init__(
        self,
        items_per_slot: int = 6,
        min_budget: float = 800.0,
        max_budget: float = 2000.0,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            items_per_slot: Number of candidate items generated per equipment slot.
            min_budget:     Minimum total budget for an instance.
            max_budget:     Maximum total budget for an instance.
            device:         Device for generated tensors.
        """
        self.items_per_slot = items_per_slot
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.device = torch.device(device)
        self._kwargs = kwargs

    @property
    def num_items(self) -> int:
        """Total items in the pool: num_slots × items_per_slot."""
        return self.num_slots * self.items_per_slot

    @property
    def stat_dim(self) -> int:
        return len(self.stat_names)

    def __call__(self, batch_size: Union[int, list, tuple] = 1) -> TensorDict:
        batch_size = (batch_size,) if isinstance(batch_size, int) else tuple(batch_size)
        return self._generate(batch_size).to(self.device)

    @abstractmethod
    def _generate(self, batch_size: tuple) -> TensorDict:
        """Generate a batch of problem instances. Must return a TensorDict."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _make_slot_ids(self, batch_size: tuple) -> torch.Tensor:
        """Create a [B, N] tensor mapping each item index to its slot index."""
        base = torch.arange(self.num_items, device=self.device) // self.items_per_slot
        return base.unsqueeze(0).expand(*batch_size, -1).clone()

    def _sample_budgets(self, batch_size: tuple) -> torch.Tensor:
        """Sample total budgets uniformly in [min_budget, max_budget]."""
        return (
            torch.rand(*batch_size, device=self.device) * (self.max_budget - self.min_budget) + self.min_budget
        )

    def _sample_rarities(self, batch_size: tuple) -> torch.Tensor:
        """
        Sample rarity multipliers.

        Rarity tiers and their multipliers mirror :class:`core.item.Rarity`:
            COMMON=1.0, UNCOMMON=1.15, RARE=1.35, EPIC=1.6, LEGENDARY=2.0

        Higher rarities are progressively less likely.
        """
        rarity_values = torch.tensor([1.0, 1.15, 1.35, 1.6, 2.0], device=self.device)
        # Weighted draw: common items are most frequent
        weights = torch.tensor([0.35, 0.30, 0.20, 0.10, 0.05], device=self.device)
        indices = torch.multinomial(
            weights.unsqueeze(0).expand(*batch_size, self.num_items, -1).reshape(-1, 5),
            num_samples=1,
        ).squeeze(-1)
        return rarity_values[indices].reshape(*batch_size, self.num_items)

    def _sample_costs(self, batch_size: tuple, rarities: torch.Tensor) -> torch.Tensor:
        """
        Sample item costs correlated with rarity and slot tier.

        Base cost in [base_min, base_max] is scaled by the rarity multiplier.
        """
        base_min, base_max = self._cost_range()
        base = torch.rand(*batch_size, self.num_items, device=self.device) * (base_max - base_min) + base_min
        return base * rarities

    def _cost_range(self):
        """(min_cost, max_cost) for a base item before rarity scaling."""
        return 30.0, 300.0

    def to(self, device: Union[str, torch.device]) -> "BuildGenerator":
        """Return a copy of the generator on the specified device."""
        cls = self.__class__
        kw = {
            "items_per_slot": self.items_per_slot,
            "min_budget": self.min_budget,
            "max_budget": self.max_budget,
            "device": device,
            **self._kwargs,
        }
        return cls(**kw)
