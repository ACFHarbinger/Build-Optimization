"""
FPS (First-Person Shooter) loadout generator.

Generates synthetic equipment instances for a tactical FPS game with 8 slots.
Each slot corresponds to a loadout category; stats reflect gunplay mechanics.

Stat names (7):
    damage, accuracy, range, mobility, reload_speed, recoil_control, magazine_size

Slots (8):
    0 PRIMARY_WEAPON   1 SECONDARY_WEAPON   2 HELMET     3 VEST
    4 GLOVES           5 BOOTS              6 TACTICAL   7 LETHAL
"""

from __future__ import annotations

from typing import Any, Union

import torch
from tensordict import TensorDict

from .base import BuildGenerator


def _profile(damage=(0, 0), accuracy=(0, 0), range_=(0, 0), mobility=(0, 0),
              reload_speed=(0, 0), recoil_control=(0, 0), magazine_size=(0, 0)):
    return torch.tensor(
        [damage, accuracy, range_, mobility, reload_speed, recoil_control, magazine_size],
        dtype=torch.float32,
    )


_SLOT_PROFILES = [
    # 0 PRIMARY_WEAPON — high damage and range; low mobility
    _profile(damage=(40, 100), accuracy=(50, 90), range_=(40, 100), mobility=(20, 60),
             reload_speed=(30, 70), recoil_control=(30, 80), magazine_size=(20, 60)),
    # 1 SECONDARY_WEAPON — fast draw, lower damage
    _profile(damage=(20, 55), accuracy=(50, 85), range_=(20, 60), mobility=(60, 100),
             reload_speed=(50, 90), recoil_control=(40, 75), magazine_size=(8, 20)),
    # 2 HELMET — pure damage-reduction / accuracy bonus
    _profile(damage=(0, 5), accuracy=(5, 20), range_=(0, 0), mobility=(0, 10),
             reload_speed=(0, 0), recoil_control=(5, 20), magazine_size=(0, 0)),
    # 3 VEST — mobility vs protection trade-off
    _profile(damage=(0, 0), accuracy=(0, 10), range_=(0, 0), mobility=(10, 50),
             reload_speed=(5, 20), recoil_control=(0, 15), magazine_size=(0, 10)),
    # 4 GLOVES — grip/handling; affects recoil and reload
    _profile(damage=(0, 5), accuracy=(5, 20), range_=(0, 0), mobility=(5, 15),
             reload_speed=(10, 35), recoil_control=(15, 40), magazine_size=(0, 0)),
    # 5 BOOTS — agility and noise reduction (mobility)
    _profile(damage=(0, 0), accuracy=(0, 5), range_=(0, 0), mobility=(20, 60),
             reload_speed=(0, 10), recoil_control=(0, 5), magazine_size=(0, 0)),
    # 6 TACTICAL (stun / flash / heartbeat) — utility-focused
    _profile(damage=(0, 20), accuracy=(10, 30), range_=(5, 30), mobility=(0, 10),
             reload_speed=(0, 0), recoil_control=(0, 0), magazine_size=(1, 4)),
    # 7 LETHAL (grenade / claymore) — burst damage, no sustained stats
    _profile(damage=(30, 90), accuracy=(0, 30), range_=(10, 50), mobility=(0, 0),
             reload_speed=(0, 0), recoil_control=(0, 0), magazine_size=(1, 3)),
]


class FPSGenerator(BuildGenerator):
    """
    Synthetic loadout generator for a tactical first-person shooter.

    Produces ``8 × items_per_slot`` items per instance with FPS-appropriate
    stat distributions.  All stats are normalised to [0, 1].
    """

    stat_names = ["damage", "accuracy", "range", "mobility", "reload_speed", "recoil_control", "magazine_size"]
    num_slots = 8

    def __init__(
        self,
        items_per_slot: int = 6,
        min_budget: float = 600.0,
        max_budget: float = 1800.0,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(items_per_slot, min_budget, max_budget, device, **kwargs)
        # Stat normalisation ceilings (raw max values per stat)
        self._stat_max = torch.tensor([100., 100., 100., 100., 90., 80., 60.], device=self.device)

    def _generate(self, batch_size: tuple) -> TensorDict:
        B = batch_size
        N = self.num_items   # 8 * items_per_slot
        S = self.stat_dim    # 7

        profiles = torch.stack(_SLOT_PROFILES, dim=0).to(self.device)   # [8, 7, 2]
        slot_profiles = profiles.unsqueeze(0).repeat_interleave(self.items_per_slot, dim=0)  # [N, 7, 2]
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
        return 40.0, 300.0
