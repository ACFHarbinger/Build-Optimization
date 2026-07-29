"""model.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import model
"""

from typing import Any, Protocol, runtime_checkable

import torch
from tensordict import TensorDict


@runtime_checkable
class IModel(Protocol):
    """
    Protocol for Neural Models.
    """

    def forward(self, td: TensorDict, **kwargs: Any) -> Any:
        """Forward.

        Args:
            td (TensorDict): Description of td.
            kwargs (Any): Description of kwargs.
        """
        ...

    def to(self, device: torch.device) -> "IModel":
        """To.

        Args:
            device (torch.device): Description of device.
        """
        ...

    def eval(self) -> "IModel":
        """Eval."""
        ...

    def train(self) -> "IModel":
        """Train."""
        ...
