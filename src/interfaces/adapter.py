"""adapter.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import adapter
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class IPolicyAdapter(ABC):
    """
    Interface for all routing policy adapters.
    Adapts various policies (Neural, Classical, Heuristic) to a common execution interface
    for the simulator.
    """

    @abstractmethod
    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the policy to generate a route.

        Args:
            **kwargs: Context dictionary containing simulation state.

        Returns:
            Tuple[List[int], float, Any]: (tour, cost, additional_output)
        """
        pass
