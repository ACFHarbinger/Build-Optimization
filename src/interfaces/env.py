"""env.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import env
"""

from typing import Any, Optional, Protocol

import torch
from tensordict import TensorDictBase


class IEnv(Protocol):
    """
    Protocol for RL Environments.
    Matches RL4COEnvBase interface.
    """

    name: str

    @property
    def device(self) -> torch.device:
        """Device to place tensors on."""
        ...

    generator: Optional[Any] = None

    def reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs: Any) -> TensorDictBase:
        """Reset environment."""
        ...

    def step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Step environment."""
        ...

    def get_reward(self, tensordict: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate reward."""
        ...

    def get_action_mask(self, tensordict: TensorDictBase) -> torch.Tensor:
        """Get valid action mask."""
        ...
