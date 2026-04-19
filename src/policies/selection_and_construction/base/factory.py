"""
Joint Policy Factory Module.

Provides a unified factory for instantiating joint selection-and-construction
solvers with their appropriate configuration dataclasses.
"""

from typing import Any, Dict, Optional, Union

from .base_joint_policy import BaseJointPolicy
from .registry import JointPolicyRegistry


class JointPolicyFactory:
    """
    Factory for creating joint selection-and-construction solvers.
    """

    _registered = False

    @classmethod
    def ensure_registered(cls) -> None:
        """Import all solver modules to trigger @JointPolicyRegistry.register() decorators."""
        if cls._registered:
            return

        # Core solvers
        import logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga as nds_brkga  # noqa
        import logic.src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa as sa  # noqa
        import logic.src.policies.selection_and_construction.joint_greedy_orienteering.policy_jgo as greedy  # noqa

        cls._registered = True

    @staticmethod
    def get_solver(
        name: str,
        config: Optional[Union[Dict[str, Any], Any]] = None,
        **kwargs: Any,
    ) -> BaseJointPolicy:
        """
        Create and return the appropriate joint solver instance.

        Args:
            name: Solver name (e.g., 'nds_brkga', 'jsa', 'jgo').
            config: Configuration object or dict.
            **kwargs: Extra arguments passed to the solver constructor.

        Returns:
            Instantiated joint solver.
        """
        JointPolicyFactory.ensure_registered()

        # Normalize name
        name = name.lower()

        cls = JointPolicyRegistry.get(name)
        if not cls:
            raise ValueError(f"Unknown joint solver: {name}. Registered solvers: {JointPolicyRegistry.list_policies()}")

        if config is not None:
            return cls(config=config, **kwargs)  # type: ignore[call-arg]
        return cls(**kwargs)  # type: ignore[call-arg]
