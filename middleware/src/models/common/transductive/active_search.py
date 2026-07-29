"""active_search.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import active_search
"""

from typing import Any

from .base import TransductiveModel


class ActiveSearch(TransductiveModel):
    """
    Active Search (Bello et al. 2016).

    Optimizes all parameters of the wrapped model on individual test instances.
    """

    def _get_search_params(self) -> Any:
        """
        Active Search optimizes all parameters.
        """
        return self.model.parameters()
