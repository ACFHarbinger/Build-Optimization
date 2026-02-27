"""policy.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import policy
"""

from typing import Any, Dict, Optional, Protocol, Union

import torch
from tensordict import TensorDict


class IPolicy(Protocol):
    """
    Protocol for RL Policies.
    Matches the interface expected by RL4COLitModule and StepwisePPO.
    """

    encoder: Optional[torch.nn.Module]
    decoder: Optional[torch.nn.Module]

    def forward(
        self,
        td: TensorDict,
        env: Optional[Any] = None,
        strategy: str = "sampling",
        num_starts: int = 1,
        **kwargs: Any,
    ) -> Union[TensorDict, Dict[str, Any]]:
        """
        Forward pass to generate solutions.
        """
        ...

    def __call__(self, *args, **kwargs) -> Any:
        """call  .

        Args:
            args (Any): Description of args.
            kwargs (Any): Description of kwargs.
        """
        ...
