"""adapter.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import adapter
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    from core.build import Build


class IPolicyAdapter(ABC):
    """
    Interface for all routing policy adapters.
    Adapts various policies (Neural, Classical, Heuristic) to a common execution interface
    for the simulator.
    """

    @abstractmethod
    def execute(self, **kwargs: Any) -> Tuple["Build", float, Any]:
        """
        Execute the policy to generate a build.

        Args:
            **kwargs: Context dictionary containing simulation state.

        Returns:
            Tuple[Build, float, Any]: (best_build, best_score, additional_output)
        """
        pass
