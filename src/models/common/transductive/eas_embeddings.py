"""eas_embeddings.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import eas_embeddings
"""

from typing import Any, Dict, Optional

from torch import nn

from .eas import EAS


class EASEmb(EAS):
    """
    EAS variant that only optimizes instance-specific embeddings (latents).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_search_steps: int = 20,
        **kwargs: Any,
    ):
        """Initialize Class.

        Args:
            model (nn.Module): Description of model.
            optimizer_kwargs (Optional[Dict[str, Any]]): Description of optimizer_kwargs.
            n_search_steps (int): Description of n_search_steps.
            kwargs (Any): Description of kwargs.
        """
        super().__init__(
            model=model,
            optimizer_kwargs=optimizer_kwargs,
            n_search_steps=n_search_steps,
            search_param_names=["init_embedding"],
            **kwargs,
        )
