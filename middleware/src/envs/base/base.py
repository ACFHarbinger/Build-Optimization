"""
Base environment class for build-optimization problems.

Mirrors WSmart-Route's RL4COEnvBase but targets the item-selection (build
optimisation) domain rather than vehicle routing.
"""

from typing import Any, Dict, List, Optional, Union

import torch
from torchrl.envs import EnvBase

from .batch import BatchMixin
from .ops import OpsMixin


class BuildEnvBase(BatchMixin, OpsMixin, EnvBase):
    """
    Base class for sequential build-optimisation environments.

    At each timestep the agent selects one item from the available pool to
    equip into its designated slot.  The episode terminates when all slots are
    filled or no valid item remains within the budget.

    Attributes:
        name:         Unique environment identifier.
        stat_names:   Ordered list of stat feature names for this game type.
        num_slots:    Number of equipment slots in a complete build.
        slot_bonus:   Reward bonus per filled slot (encourages completion).
        rarity_bonus: Reward bonus per unit of rarity for each equipped item.
    """

    name: str = "build"
    stat_names: List[str] = []
    num_slots: int = 0

    def __init__(
        self,
        generator: Optional[Any] = None,
        generator_params: Optional[dict] = None,
        stat_weights: Optional[Dict[str, float]] = None,
        slot_bonus: float = 5.0,
        rarity_bonus: float = 2.0,
        device: Union[str, torch.device] = "cpu",
        batch_size: Optional[Union[torch.Size, int]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            generator:        Instance generator (creates TensorDict problem instances).
            generator_params: Keyword args forwarded to the generator if not provided.
            stat_weights:     Per-stat reward multipliers keyed by stat name.
                              Missing stats default to 1.0.
            slot_bonus:       Additive reward per filled slot.
            rarity_bonus:     Additive reward per rarity unit of equipped items.
            device:           Torch device.
            batch_size:       Environment batch size.
        """
        if batch_size is None:
            batch_size = torch.Size([])
        elif isinstance(batch_size, int):
            batch_size = torch.Size([batch_size])
        else:
            batch_size = torch.Size(batch_size)

        env_base_kwargs: Dict[str, Any] = {"device": device, "batch_size": batch_size}
        for k in ["run_type_checks", "allow_done_after_reset"]:
            if k in kwargs:
                env_base_kwargs[k] = kwargs.pop(k)

        super().__init__(**env_base_kwargs)
        self.check_env_specs = kwargs.get("check_env_specs", False)

        self.generator = generator
        self.generator_params = generator_params or {}
        self.slot_bonus = slot_bonus
        self.rarity_bonus = rarity_bonus

        # Build stat weight tensor (aligned to self.stat_names order)
        weights = stat_weights or {}
        weight_values = [weights.get(s, 1.0) for s in self.stat_names]
        self._stat_weights = torch.tensor(weight_values, dtype=torch.float32, device=device)

    def render(self, td: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"Rendering not implemented for {self.name}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, device={self.device})"


# Inherited RL4CO policy/model modules still import this name.
RL4COEnvBase = BuildEnvBase
