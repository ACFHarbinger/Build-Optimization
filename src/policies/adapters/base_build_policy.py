"""
Base Build Policy Module.

Provides a template base class for build optimization policies, extracting common
functionality like parameter loading.
"""

from abc import abstractmethod
from dataclasses import fields
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type

if TYPE_CHECKING:
    from core.build import Build

import numpy as np

from core.problem import BuildProblem
from interfaces.adapter import IPolicyAdapter


def _flatten_raw_config(source: Any) -> Dict[str, Any]:
    """Flatten nested config structures."""
    result: Dict[str, Any] = {}
    if isinstance(source, list):
        for item in source:
            result.update(_flatten_raw_config(item))
    elif hasattr(source, "items"):
        for k, v in source.items():
            if k in ("custom", "params") and isinstance(v, (dict, list)):
                result.update(_flatten_raw_config(v))
            else:
                result[k] = v
    return result


class BaseBuildPolicy(IPolicyAdapter):
    """
    Base class for build optimization policies.

    Subclasses implement `_run_solver()` and `_get_config_key()`.
    """

    def __init__(self, config: Any = None):
        """Initialize policy with optional config dataclass or raw dict."""
        if config is not None and isinstance(config, dict):
            self._config = self._build_config(config)
        else:
            self._config = config

    @property
    def config(self) -> Any:
        return self._config

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return None

    @classmethod
    def _build_config(cls, raw_config: Dict[str, Any]) -> Any:
        config_cls = cls._config_class()
        if config_cls is None:
            return None

        config_key = cls._get_config_key(cls)  # type: ignore[arg-type]
        policy_section = raw_config.get(config_key, raw_config)

        flat = _flatten_raw_config(policy_section)
        valid_fields = {f.name for f in fields(config_cls)}
        filtered = {k: v for k, v in flat.items() if k in valid_fields}

        return config_cls(**filtered)

    def _get_config_key(self) -> str:
        return "default"

    @abstractmethod
    def _run_solver(
        self,
        problem: BuildProblem,
        budget: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float]:
        """
        Run the specific solver for this policy.

        Returns:
            Tuple of (build_array, score)
        """
        pass

    def execute(self, **kwargs: Any) -> Tuple["Build", float, Any]:
        """
        Execute the build policy.

        Args:
            **kwargs: Must contain `problem` (BuildProblem) and `budget` (float).

        Returns:
            Tuple of (best_build, cost, dict) - Wait, we should return a core.build.Build.
            But the solvers return np.ndarray. We map it back here!
        """
        problem = kwargs["problem"]
        budget = kwargs["budget"]
        config = kwargs.get("config", {})

        # Build the policy config dict
        if self._config is not None:
            config_key = self._get_config_key()
            runtime_overrides = config.get(config_key, {})
            if not isinstance(runtime_overrides, dict):
                runtime_overrides = (
                    _flatten_raw_config(runtime_overrides) if hasattr(runtime_overrides, "items") else {}
                )
            from dataclasses import asdict

            policy_config = {**asdict(self._config), **runtime_overrides}
        else:
            config_key = self._get_config_key()
            policy_section = config.get(config_key, config)
            policy_config = _flatten_raw_config(policy_section) if hasattr(policy_section, "items") else {}

        # Run solver
        build_arr, score = self._run_solver(
            problem=problem,
            budget=budget,
            values=policy_config,
            **kwargs,
        )

        # Map back to core.build.Build
        from core.build import Build

        # We need the original Items to reconstruct the build.
        # But wait, problem only has numpy arrays!
        # For now, since the interface requires a Build, we might need the original items list in kwargs.
        items_list = kwargs.get("items_list")
        final_build = Build(budget=budget)

        if items_list and build_arr is not None:
            for item_idx in build_arr:
                if item_idx != -1:
                    final_build.equip(items_list[item_idx])

        return final_build, score, {}
