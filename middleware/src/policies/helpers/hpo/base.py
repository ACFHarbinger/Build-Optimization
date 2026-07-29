"""
Base classes for simulation policy HPO.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import optuna
from logic.src.configs import Config


class PolicyHPOBase(ABC):
    """Abstract base class for simulation policy Hyperparameter Optimization."""

    def __init__(self, cfg: Config):
        """Initialize PolicyHPO.

        Args:
            cfg: Root application configuration.
        """
        self.cfg = cfg
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: float = float("-inf")

    @abstractmethod
    def run(self, n_trials: int = 20) -> float:
        """Run the HPO process.

        Args:
            n_trials: Number of optimization trials.

        Returns:
            The best metric value found.
        """
        pass

    def _apply_params(self, cfg: Config, params: Dict[str, Any]) -> None:
        """Apply sampled parameters to the config.

        Args:
            cfg: Config object to mutate.
            params: Dictionary of sampled parameters.
        """
        for key, value in params.items():
            # Support nested keys like 'sim.full_policies.0.alns.max_iterations'
            parts = key.split(".")
            target = cfg
            for part in parts[:-1]:
                if hasattr(target, part):
                    target = getattr(target, part)
                elif isinstance(target, dict) and part in target:
                    target = target[part]
                elif isinstance(target, list) and part.isdigit():
                    target = target[int(part)]
                else:
                    # Handle omegaconf ListConfig/DictConfig if needed
                    try:
                        target = target[part] if not part.isdigit() else target[int(part)]  # type: ignore[index]
                    except (KeyError, IndexError, TypeError) as err:
                        raise AttributeError(f"Cannot find config path: {key}") from err

            last_part = parts[-1]
            if hasattr(target, last_part):
                setattr(target, last_part, value)
            else:
                target[last_part] = value  # type: ignore[index]

    @staticmethod
    def suggest_param(trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> Any:
        """Suggest a parameter based on its specification.

        Args:
            trial: Optuna trial object.
            name: Name of the parameter.
            spec: Specification dictionary (type, low, high, choices, etc.).

        Returns:
            The suggested parameter value.
        """
        p_type = spec.get("type")
        if p_type == "float":
            return trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step"),
                log=spec.get("log", False),
            )
        elif p_type == "int":
            return trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
                log=spec.get("log", False),
            )
        elif p_type == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown parameter type: {p_type}")
