"""
Core operations mixin for build optimization environments.

Analogous to WSmart-Route's OpsMixin but adapted for item selection (build
optimization) instead of node routing. The episode is a sequential slot-filling
process: at each step the agent selects one item from the available pool.
"""

from abc import abstractmethod
from typing import Any, Optional, cast

import torch
from tensordict import TensorDictBase


class OpsMixin:
    """Mixin for step, reset, and score calculations in build-optimization envs."""

    def _make_spec(self, generator: Optional[Any] = None) -> None:
        from torchrl.data import DiscreteTensorSpec, TensorSpec, UnboundedContinuousTensorSpec

        batch_size = getattr(self, "batch_size", torch.Size([]))
        device = getattr(self, "device", "cpu")
        self.done_spec = cast(
            TensorSpec,
            DiscreteTensorSpec(n=2, shape=torch.Size((*batch_size, 1)), dtype=torch.bool, device=device),
        )
        self.reward_spec = cast(
            TensorSpec,
            UnboundedContinuousTensorSpec(shape=torch.Size((*batch_size, 1)), device=device),
        )

    def step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.batch_size = tensordict.batch_size
        return cast(TensorDictBase, super().step(tensordict))  # type: ignore[misc]

    def reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDictBase:
        if tensordict is None:
            tensordict = kwargs.pop("tensordict", None)
            if tensordict is None:
                tensordict = kwargs.pop("td", None)
        if tensordict is not None:
            self.batch_size = tensordict.batch_size
        return self._reset(tensordict, **kwargs)

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDictBase:
        """Initialise episode state from a problem instance (generated or provided)."""
        batch_size_arg = kwargs.pop("batch_size", None)
        if tensordict is None:
            if self.generator is None:  # type: ignore[attr-defined]
                raise ValueError("Either provide tensordict or set a generator for the environment")
            tensordict = self.generator(batch_size_arg or self.batch_size)  # type: ignore[attr-defined]
        else:
            tensordict = tensordict.clone()

        assert tensordict is not None
        tensordict = tensordict.to(self.device)  # type: ignore[attr-defined]
        tensordict = self._reset_instance(tensordict)

        tensordict["action_mask"] = self._get_action_mask(tensordict)
        tensordict["i"] = torch.zeros(tensordict.batch_size, dtype=torch.long, device=self.device)  # type: ignore[attr-defined]
        tensordict["done"] = torch.zeros((*tensordict.batch_size, 1), dtype=torch.bool, device=self.device)  # type: ignore[attr-defined]

        if "terminated" in tensordict:
            tensordict["terminated"] = torch.zeros((*tensordict.batch_size, 1), dtype=torch.bool, device=self.device)  # type: ignore[attr-defined]
        if "truncated" in tensordict:
            tensordict["truncated"] = torch.zeros((*tensordict.batch_size, 1), dtype=torch.bool, device=self.device)  # type: ignore[attr-defined]

        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute an item-equip action and return the next state."""
        td_next = tensordict.copy()
        for key in ["next", "reward", "done"]:
            if key in td_next.keys():
                del td_next[key]

        td_next = self._step_instance(td_next)

        td_next["i"] = tensordict["i"] + 1
        td_next["action_mask"] = self._get_action_mask(td_next)
        done = self._check_done(td_next)

        if done.dim() == len(td_next.batch_size):
            done = done.unsqueeze(-1)
        elif done.dim() > len(td_next.batch_size) + 1:
            done = done.reshape((*td_next.batch_size, 1))

        td_next["done"] = done
        td_next["terminated"] = done.clone()
        td_next["truncated"] = torch.zeros_like(done)

        reward = self._get_reward(td_next, tensordict.get("action", None))
        if reward.dim() == len(td_next.batch_size):
            reward = reward.unsqueeze(-1)
        elif reward.dim() > len(td_next.batch_size) + 1:
            reward = reward.reshape((*td_next.batch_size, 1))
        td_next["reward"] = reward

        return td_next

    def _step_instance(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Core transition: equip the selected item and update build state.

        Updates:
          - ``equipped``         [B, N] bool  — marks selected item as equipped
          - ``slot_filled``      [B, S] bool  — marks the item's slot as occupied
          - ``budget_remaining`` [B]          — deducts item cost
          - ``prev_score``       [B]          — saves score before this step
          - ``current_score``    [B]          — recomputes score after equip
        """
        action = tensordict["action"]  # [B]
        if action.dim() > 1:
            action = action.squeeze(-1)

        # Snapshot score before equipping (used for dense reward delta)
        tensordict["prev_score"] = tensordict["current_score"].clone()

        # Equip item
        equipped = tensordict["equipped"].clone()
        equipped.scatter_(1, action.unsqueeze(-1), True)
        tensordict["equipped"] = equipped

        # Mark slot as filled
        slot_ids = tensordict["slot_ids"]  # [B, N] or [1, N]
        action_slot = slot_ids.gather(1, action.unsqueeze(-1)).squeeze(-1)  # [B]
        slot_filled = tensordict["slot_filled"].clone()
        slot_filled.scatter_(1, action_slot.unsqueeze(-1), True)
        tensordict["slot_filled"] = slot_filled

        # Deduct cost
        action_cost = tensordict["costs"].gather(1, action.unsqueeze(-1)).squeeze(-1)  # [B]
        tensordict["budget_remaining"] = tensordict["budget_remaining"] - action_cost

        # Recompute build score
        tensordict["current_score"] = self._compute_score(tensordict)

        return tensordict

    def _compute_score(self, tensordict: TensorDictBase) -> torch.Tensor:
        """
        Vectorised build score: weighted stats + slot-fill bonus + rarity bonus.

        This default implementation is overridable by game-specific subclasses.
        """
        equipped = tensordict["equipped"]          # [B, N]
        items = tensordict["items"]                # [B, N, S]
        rarities = tensordict["rarities"]          # [B, N]
        slot_filled = tensordict["slot_filled"]    # [B, num_slots]

        # Stat contribution
        stat_weights = getattr(self, "_stat_weights", None)  # type: ignore[attr-defined]
        if stat_weights is None:
            stat_weights = torch.ones(items.shape[-1], device=items.device)
        equipped_stats = (items * equipped.unsqueeze(-1)).sum(dim=1)  # [B, S]
        stat_score = (equipped_stats * stat_weights.to(items.device)).sum(-1)  # [B]

        # Slot-fill bonus
        slot_bonus: float = getattr(self, "slot_bonus", 5.0)  # type: ignore[attr-defined]
        fill_score = slot_filled.float().sum(-1) * slot_bonus  # [B]

        # Rarity bonus
        rarity_bonus: float = getattr(self, "rarity_bonus", 2.0)  # type: ignore[attr-defined]
        rarity_score = (rarities * equipped.float()).sum(-1) * rarity_bonus  # [B]

        return stat_score + fill_score + rarity_score

    def _check_done(self, tensordict: TensorDictBase) -> torch.Tensor:
        """
        Episode ends when all slots are filled or no valid action remains.
        """
        slot_filled = tensordict["slot_filled"]  # [B, num_slots]
        action_mask = tensordict["action_mask"]  # [B, N]

        all_slots_filled = slot_filled.all(dim=-1)  # [B]
        no_valid_action = ~action_mask.any(dim=-1)   # [B]
        done = all_slots_filled | no_valid_action

        try:
            return done.reshape(tensordict.batch_size)
        except Exception:
            return done.flatten().reshape(tensordict.batch_size)

    def get_reward(self, tensordict: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._get_reward(tensordict, actions)

    def get_costs(
        self,
        tensordict: TensorDictBase,
        pi: torch.Tensor,
        **kwargs: Any,
    ) -> Any:
        """Follow a sequence of actions and return negative reward (cost)."""
        curr_td = self.reset(tensordict)
        for i in range(pi.size(1)):
            curr_td["action"] = pi[:, i]
            curr_td = self.step(curr_td)["next"]
        reward = self.get_reward(curr_td, pi)
        return -reward, {"total": -reward}, curr_td

    @abstractmethod
    def _reset_instance(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError

    @abstractmethod
    def _get_reward(self, tensordict: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _get_action_mask(self, tensordict: TensorDictBase) -> torch.Tensor:
        raise NotImplementedError

    def _set_seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            torch.manual_seed(seed)
