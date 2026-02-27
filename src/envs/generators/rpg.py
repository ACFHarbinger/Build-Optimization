"""
RPG item-pool generator.

Generates synthetic equipment instances for a fantasy RPG with 10 equipment
slots.  Each slot has a characteristic stat profile that reflects real RPG item
design (e.g. weapons have high attack, helmets have high defence/health).

Stat names (7):
    attack, defense, magic_power, health, speed, critical_rate, critical_damage

Slots (10):
    0 WEAPON   1 HELMET   2 CHEST    3 GLOVES   4 BOOTS
    5 RING_1   6 RING_2   7 AMULET   8 ACCESSORY_1   9 ACCESSORY_2
"""

from __future__ import annotations

from typing import Any, Union

import torch
from tensordict import TensorDict

from .base import BuildGenerator

# Stat index constants
_ATK, _DEF, _MGP, _HP, _SPD, _CR, _CD = 0, 1, 2, 3, 4, 5, 6


def _profile(attack=(0, 0), defense=(0, 0), magic_power=(0, 0), health=(0, 0),
              speed=(0, 0), critical_rate=(0, 0), critical_damage=(0, 0)):
    """Return a (7, 2) tensor of (min, max) stat ranges in order."""
    return torch.tensor([attack, defense, magic_power, health, speed, critical_rate, critical_damage],
                        dtype=torch.float32)


# Per-slot stat distributions: each row is (min, max) for that stat
_SLOT_PROFILES = [
    # 0 WEAPON
    _profile(attack=(30, 120), defense=(0, 10), magic_power=(0, 50),
             health=(0, 0), speed=(0, 10), critical_rate=(0, 15), critical_damage=(0, 25)),
    # 1 HELMET
    _profile(attack=(0, 15), defense=(15, 60), magic_power=(0, 25),
             health=(20, 80), speed=(0, 0), critical_rate=(0, 8), critical_damage=(0, 0)),
    # 2 CHEST
    _profile(attack=(0, 15), defense=(20, 80), magic_power=(0, 30),
             health=(30, 100), speed=(0, 5), critical_rate=(0, 10), critical_damage=(0, 0)),
    # 3 GLOVES
    _profile(attack=(5, 40), defense=(10, 35), magic_power=(0, 20),
             health=(0, 25), speed=(0, 5), critical_rate=(0, 12), critical_damage=(0, 15)),
    # 4 BOOTS
    _profile(attack=(0, 10), defense=(10, 45), magic_power=(0, 0),
             health=(15, 50), speed=(10, 30), critical_rate=(0, 5), critical_damage=(0, 0)),
    # 5 RING_1
    _profile(attack=(5, 30), defense=(0, 12), magic_power=(0, 25),
             health=(10, 35), speed=(0, 8), critical_rate=(5, 10), critical_damage=(0, 12)),
    # 6 RING_2
    _profile(attack=(5, 25), defense=(5, 12), magic_power=(0, 25),
             health=(10, 35), speed=(0, 8), critical_rate=(0, 8), critical_damage=(0, 8)),
    # 7 AMULET
    _profile(attack=(5, 25), defense=(10, 30), magic_power=(20, 45),
             health=(30, 60), speed=(0, 10), critical_rate=(0, 10), critical_damage=(0, 15)),
    # 8 ACCESSORY_1
    _profile(attack=(0, 20), defense=(10, 40), magic_power=(10, 25),
             health=(10, 50), speed=(0, 0), critical_rate=(0, 5), critical_damage=(0, 0)),
    # 9 ACCESSORY_2
    _profile(attack=(5, 18), defense=(0, 15), magic_power=(0, 15),
             health=(0, 20), speed=(5, 10), critical_rate=(5, 12), critical_damage=(5, 10)),
]


class RPGGenerator(BuildGenerator):
    """
    Synthetic item-pool generator for fantasy RPG builds.

    Produces ``num_slots × items_per_slot = 10 × items_per_slot`` items per
    instance.  Stat values are sampled from slot-specific uniform ranges and
    normalised to [0, 1] relative to the max possible value per stat.
    """

    stat_names = ["attack", "defense", "magic_power", "health", "speed", "critical_rate", "critical_damage"]
    num_slots = 10

    def __init__(
        self,
        items_per_slot: int = 6,
        min_budget: float = 800.0,
        max_budget: float = 2000.0,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(items_per_slot, min_budget, max_budget, device, **kwargs)

        # Maximum possible value per stat (for normalisation)
        self._stat_max = torch.tensor([120., 80., 50., 100., 30., 15., 25.], device=self.device)

    def _generate(self, batch_size: tuple) -> TensorDict:
        B = batch_size
        N = self.num_items   # 10 * items_per_slot
        S = self.stat_dim    # 7

        # Gather per-slot profiles as [num_slots, S, 2] (min/max)
        profiles = torch.stack(_SLOT_PROFILES, dim=0).to(self.device)  # [10, 7, 2]

        # Expand to [B, N, S] by repeating each slot's profile items_per_slot times
        slot_profiles = profiles.unsqueeze(0).repeat_interleave(self.items_per_slot, dim=0)  # [N, S, 2]
        # Broadcast across batch
        lo = slot_profiles[None, :, :, 0].expand(*B, N, S)  # [B, N, S]
        hi = slot_profiles[None, :, :, 1].expand(*B, N, S)  # [B, N, S]

        # Sample raw stats
        raw = torch.rand(*B, N, S, device=self.device) * (hi - lo) + lo  # [B, N, S]

        # Normalise to [0, 1]
        items = raw / self._stat_max.view(1, 1, S).to(self.device)
        items = items.clamp(0.0, 1.0)

        # Rarity and costs
        rarities = self._sample_rarities(B)                   # [B, N]
        costs = self._sample_costs(B, rarities)               # [B, N]
        slot_ids = self._make_slot_ids(B)                     # [B, N]
        budget = self._sample_budgets(B)                      # [B]

        return TensorDict(
            {
                "items": items,
                "costs": costs,
                "slot_ids": slot_ids,
                "rarities": rarities,
                "budget": budget,
            },
            batch_size=B,
            device=self.device,
        )

    def _cost_range(self):
        return 50.0, 400.0
