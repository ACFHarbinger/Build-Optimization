from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union

from .environment_tags import EnvironmentTag
from .model_tags import ModelTag
from .operator_tags import OperatorTag
from .policy_tags import PolicyTag
from .trainer_tags import TrainerTag

# The Unified Type Alias
AnyTag = Union[PolicyTag, ModelTag, EnvironmentTag, OperatorTag, TrainerTag]

# Type alias for clarity: can be a Class Type or a Function
T_Algorithm = Union[Type, Callable[..., Any]]
T = TypeVar("T")


class GlobalRegistry:
    _registry: Dict[T_Algorithm, Set[AnyTag]] = {}

    @classmethod
    def register(cls, *tags: AnyTag) -> Callable[[T_Algorithm], T_Algorithm]:
        """Universal decorator for any class or function."""

        def decorator(obj: T_Algorithm) -> T_Algorithm:
            cls._registry[obj] = set(tags)
            return obj

        return decorator

    @classmethod
    def get_name(cls, obj: T_Algorithm) -> str:
        """Helper to get a readable name for either a class or a function."""
        return getattr(obj, "__name__", str(obj))

    @classmethod
    def query_intersection(
        cls, *tags: AnyTag, expected_type: Optional[Type[T]] = None
    ) -> Union[List[T_Algorithm], List[Type[T]]]:
        """
        Returns objects matching ALL tags.
        If expected_type is provided, strictly filters out non-matching classes.
        """
        query_set = set(tags)
        results: List[Any] = []

        for obj, obj_tags in cls._registry.items():
            if query_set.issubset(obj_tags):
                # If the user only wants specific Base Classes (e.g., nn.Module)
                if expected_type is not None:
                    if isinstance(obj, type) and issubclass(obj, expected_type):
                        results.append(obj)
                else:
                    results.append(obj)

        return results

    @classmethod
    def query_union(
        cls, *tags: AnyTag, expected_type: Optional[Type[T]] = None
    ) -> Union[List[T_Algorithm], List[Type[T]]]:
        """
        Returns objects matching AT LEAST ONE tag.
        If expected_type is provided, strictly filters out non-matching classes.
        """
        query_set = set(tags)
        results: List[Any] = []

        for obj, obj_tags in cls._registry.items():
            if not query_set.isdisjoint(obj_tags):
                # If the user only wants specific Base Classes (e.g., nn.Module)
                if expected_type is not None:
                    if isinstance(obj, type) and issubclass(obj, expected_type):
                        results.append(obj)
                else:
                    results.append(obj)

        return results

    @classmethod
    def query_difference(
        cls, require: List[AnyTag], exclude: List[AnyTag], expected_type: Optional[Type[T]] = None
    ) -> Union[List[T_Algorithm], List[Type[T]]]:
        """Returns algorithms that contain ALL required tags, but NO excluded tags."""
        require_set = set(require)
        exclude_set = set(exclude)
        results: List[Any] = []
        for obj, obj_tags in cls._registry.items():
            if require_set.issubset(obj_tags) and exclude_set.isdisjoint(obj_tags):
                # If the user only wants specific Base Classes (e.g., nn.Module)
                if expected_type is not None:
                    if isinstance(obj, type) and issubclass(obj, expected_type):
                        results.append(obj)
                else:
                    results.append(obj)

        return results

    @classmethod
    def get_all(cls) -> Dict[T_Algorithm, Set[AnyTag]]:
        return cls._registry
