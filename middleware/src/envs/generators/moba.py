"""
MOBA (Multiplayer Online Battle Arena) item generator.

In most MOBAs a character purchases up to 6 items that can occupy any item
slot.  Here we model 6 generic item slots (ITEM_1 … ITEM_6) each with its own
candidate pool, reflecting that different role archetypes prioritise different
item "categories" (damage, tank, support).

Stat names (8):
    attack_damage, ability_power, armor, magic_resistance,
    health, mana, cooldown_reduction, attack_speed

Slots (6):
    0 ITEM_1   1 ITEM_2   2 ITEM_3   3 ITEM_4   4 ITEM_5   5 ITEM_6

Each slot pool spans multiple item "categories" so the generator produces
a realistic mix of damage, tank, utility, and support items across slots.
"""

from __future__ import annotations

from typing import Any, Union

import torch
from tensordict import TensorDict

from .base import BuildGenerator


def _profile(
    attack_damage=(0, 0), ability_power=(0, 0), armor=(0, 0), magic_resistance=(0, 0),
    health=(0, 0), mana=(0, 0), cooldown_reduction=(0, 0), attack_speed=(0, 0),
):
    return torch.tensor(
        [attack_damage, ability_power, armor, magic_resistance,
         health, mana, cooldown_reduction, attack_speed],
        dtype=torch.float32,
    )


# MOBA items are less slot-specific than RPG — each slot can hold many archetypes.
# We vary the profile slightly per slot to model the common MOBA build progression
# (early-game utility → mid-game power spikes → late-game luxury items).
_SLOT_PROFILES = [
    # ITEM_1 — starter/component item (low cost, hybrid stats)
    _profile(attack_damage=(10, 40), ability_power=(10, 40), armor=(0, 20),
             magic_resistance=(0, 15), health=(50, 150), mana=(20, 80),
             cooldown_reduction=(0, 5), attack_speed=(0, 10)),
    # ITEM_2 — core damage item
    _profile(attack_damage=(30, 80), ability_power=(40, 100), armor=(0, 30),
             magic_resistance=(0, 20), health=(100, 300), mana=(30, 120),
             cooldown_reduction=(5, 20), attack_speed=(5, 25)),
    # ITEM_3 — defensive / tank item
    _profile(attack_damage=(0, 30), ability_power=(0, 40), armor=(30, 80),
             magic_resistance=(20, 60), health=(200, 500), mana=(0, 60),
             cooldown_reduction=(5, 15), attack_speed=(0, 10)),
    # ITEM_4 — utility / support item
    _profile(attack_damage=(0, 20), ability_power=(20, 70), armor=(10, 40),
             magic_resistance=(10, 40), health=(100, 350), mana=(50, 200),
             cooldown_reduction=(10, 30), attack_speed=(0, 15)),
    # ITEM_5 — power spike / carry item
    _profile(attack_damage=(40, 100), ability_power=(50, 120), armor=(0, 20),
             magic_resistance=(0, 15), health=(0, 200), mana=(0, 100),
             cooldown_reduction=(0, 20), attack_speed=(10, 40)),
    # ITEM_6 — luxury / situational item (high variance)
    _profile(attack_damage=(20, 90), ability_power=(20, 110), armor=(20, 70),
             magic_resistance=(15, 60), health=(100, 500), mana=(50, 200),
             cooldown_reduction=(10, 30), attack_speed=(0, 30)),
]


class MOBAGenerator(BuildGenerator):
    """
    Synthetic item generator for MOBA-style build optimisation.

    Produces ``6 × items_per_slot`` items per instance.  Items span damage,
    tank, utility, and support archetypes, and stats are normalised to [0, 1].
    """

    stat_names = [
        "attack_damage", "ability_power", "armor", "magic_resistance",
        "health", "mana", "cooldown_reduction", "attack_speed",
    ]
    num_slots = 6

    def __init__(
        self,
        items_per_slot: int = 8,
        min_budget: float = 2500.0,
        max_budget: float = 6000.0,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(items_per_slot, min_budget, max_budget, device, **kwargs)
        # Raw stat ceilings for normalisation
        self._stat_max = torch.tensor(
            [100., 120., 80., 60., 500., 200., 30., 40.], device=self.device
        )

    def _generate(self, batch_size: tuple) -> TensorDict:
        B = batch_size
        N = self.num_items   # 6 * items_per_slot
        S = self.stat_dim    # 8

        profiles = torch.stack(_SLOT_PROFILES, dim=0).to(self.device)   # [6, 8, 2]
        slot_profiles = profiles.unsqueeze(0).repeat_interleave(self.items_per_slot, dim=0)  # [N, 8, 2]
        lo = slot_profiles[None, :, :, 0].expand(*B, N, S)
        hi = slot_profiles[None, :, :, 1].expand(*B, N, S)

        raw = torch.rand(*B, N, S, device=self.device) * (hi - lo) + lo
        items = (raw / self._stat_max.view(1, 1, S).to(self.device)).clamp(0.0, 1.0)

        rarities = self._sample_rarities(B)
        costs = self._sample_costs(B, rarities)
        slot_ids = self._make_slot_ids(B)
        budget = self._sample_budgets(B)

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
        return 300.0, 1500.0
