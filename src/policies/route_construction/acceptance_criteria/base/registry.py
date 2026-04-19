from typing import Callable, Dict, List, Optional, Type

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion


class AcceptanceCriterionRegistry:
    """
    Central registry for move acceptance criteria.

    Allows for the modular decoupling of search strategies (e.g., SA, Late Acceptance)
    from the solvers that use them.
    """

    _registry: Dict[str, Type[IAcceptanceCriterion]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator to register an acceptance criterion class.

        Args:
            name: Unique identifier for the criterion (e.g., 'boltzmann', 'demon').

        Returns:
            Callable: The decorator function.
        """

        def decorator(criterion_cls: Type[IAcceptanceCriterion]) -> Type[IAcceptanceCriterion]:
            cls._registry[name.lower()] = criterion_cls
            return criterion_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type[IAcceptanceCriterion]]:
        """
        Retrieve a criterion class by name.

        Args:
            name: The name of the criterion to retrieve.

        Returns:
            Optional[Type[IAcceptanceCriterion]]: The class if found, else None.
        """
        return cls._registry.get(name.lower())

    @classmethod
    def list_criteria(cls) -> List[str]:
        """
        List all registered criteria identifiers.

        Returns:
            List[str]: A list of all registered names.
        """
        return list(cls._registry.keys())
