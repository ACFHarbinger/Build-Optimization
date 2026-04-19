"""eas.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import eas
"""

from typing import Any, Dict, List, Optional

from torch import nn

from .base import TransductiveModel


class EAS(TransductiveModel):
    """
    Efficient Active Search (Hottung et al. 2022).

    Optimizes only specific parameters (e.g., embeddings or specific layers)
    at test time to be more efficient than full Active Search.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_search_steps: int = 20,
        search_param_names: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Initialize Class.

        Args:
            model (nn.Module): Description of model.
            optimizer_kwargs (Optional[Dict[str, Any]]): Description of optimizer_kwargs.
            n_search_steps (int): Description of n_search_steps.
            search_param_names (Optional[List[str]]): Description of search_param_names.
            kwargs (Any): Description of kwargs.
        """
        super().__init__(
            model=model,
            optimizer_kwargs=optimizer_kwargs,
            n_search_steps=n_search_steps,
            **kwargs,
        )
        # Default to embeddings and projections if not specified
        self.search_param_names = search_param_names or ["init_embedding", "init_proj"]

    def _get_search_params(self) -> Any:
        """
        Identify parameters by name for optimization.
        """
        params = []
        for name, param in self.model.named_parameters():
            if any(p_name in name for p_name in self.search_param_names):
                params.append(param)

        # If no params found, fallback to all (not ideal for EAS but safe)
        if not params:
            return self.model.parameters()

        return params
