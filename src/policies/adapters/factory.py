"""
Policy Adapter module - Unified interface for all routing policies.

This module implements the Adapter design pattern to provide a consistent
interface for executing diverse routing policies within the simulator.

Now also includes the IPolicy interface and PolicyRegistry.
"""

from typing import Any, Optional

# --- IPolicy Interface ---
from logic.src.interfaces.adapter import IPolicyAdapter
from logic.src.policies.adapters.registry import PolicyRegistry

# Alias for backward compatibility
IPolicy = IPolicyAdapter


class PolicyFactory:
    """
    Factory for creating policy adapters.
    """

    _registered = False

    @classmethod
    def ensure_registered(cls) -> None:
        """Import all adapter modules to trigger @PolicyRegistry.register() decorators."""
        if cls._registered:
            return
        import logic.src.policies.neural_agent as neural_agent  # noqa
        import logic.src.policies.adapters.policy_neural as policy_neural  # noqa
        import logic.src.policies.adapters.policy_alns as policy_alns  # noqa
        import logic.src.policies.adapters.policy_bcp as policy_bcp  # noqa
        import logic.src.policies.adapters.policy_cvrp as policy_cvrp  # noqa
        import logic.src.policies.adapters.policy_hgs as policy_hgs  # noqa
        import logic.src.policies.adapters.policy_hgs_alns as policy_hgs_alns  # noqa
        import logic.src.policies.adapters.policy_lkh as policy_lkh  # noqa
        import logic.src.policies.adapters.policy_sans as policy_sans  # noqa
        import logic.src.policies.adapters.policy_tsp as policy_tsp  # noqa
        import logic.src.policies.adapters.policy_vrpp as policy_vrpp  # noqa
        import logic.src.policies.adapters.policy_ks_aco as policy_ks_aco  # noqa
        import logic.src.policies.adapters.policy_hh_aco as policy_hh_aco  # noqa
        import logic.src.policies.adapters.policy_sisr as policy_sisr  # noqa
        import logic.src.policies.adapters.policy_hvpl as policy_hvpl  # noqa
        import logic.src.policies.adapters.policy_ahvpl as policy_ahvpl  # noqa
        import logic.src.policies.adapters.policy_qde as policy_qde  # noqa
        import logic.src.policies.adapters.policy_psoma as policy_psoma  # noqa
        import logic.src.policies.adapters.policy_abc as policy_abc  # noqa
        import logic.src.policies.adapters.policy_fa as policy_fa  # noqa
        import logic.src.policies.adapters.policy_sca as policy_sca  # noqa
        import logic.src.policies.adapters.policy_hs as policy_hs  # noqa
        import logic.src.policies.adapters.policy_slc as policy_slc  # noqa
        import logic.src.policies.adapters.policy_lca as policy_lca  # noqa
        import logic.src.policies.adapters.policy_gphh as policy_gphh  # noqa
        import logic.src.policies.adapters.policy_hmm_gd as policy_hmm_gd  # noqa
        import logic.src.policies.adapters.policy_lahc as policy_lahc  # noqa
        import logic.src.policies.adapters.policy_rrt as policy_rrt  # noqa
        import logic.src.policies.adapters.policy_oba as policy_oba  # noqa
        import logic.src.policies.adapters.policy_gls as policy_gls  # noqa
        import logic.src.policies.adapters.policy_rts as policy_rts  # noqa
        import logic.src.policies.adapters.policy_ga as policy_ga  # noqa
        import logic.src.policies.adapters.policy_sa as policy_sa  # noqa
        import logic.src.policies.adapters.policy_ils as policy_ils  # noqa
        import logic.src.policies.adapters.policy_vns as policy_vns  # noqa

        cls._registered = True

    @staticmethod
    def get_adapter(
        name: str,
        config: Optional[dict] = None,
        engine: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> IPolicy:
        """
        Create and return the appropriate PolicyAdapter for the given parameters.

        Args:
            name: Policy name (e.g., 'alns', 'hgs', 'tsp').
            config: Raw policy config dict from YAML. If provided, the adapter's
                    typed config dataclass is built automatically.
            engine: Deprecated. Engine should be specified in config.
            threshold: Deprecated. Threshold should be specified in config.
            **kwargs: Additional keyword arguments (unused, for backward compat).

        Returns:
            Instantiated policy adapter with typed config.
        """
        PolicyFactory.ensure_registered()

        # Normalize name
        if not isinstance(name, str):
            raise TypeError(f"Policy name must be a string, got {type(name)}")
        name = name.lower()

        # Try Registry first
        cls = PolicyRegistry.get(name) or PolicyRegistry.get(f"policy_{name}")

        if cls:
            if config is not None:
                return cls(config=config)  # type: ignore[return-value,call-arg]
            return cls()  # type: ignore[return-value]

        raise ValueError(f"Unknown policy: {name}. Ensure it is registered in PolicyRegistry.")
