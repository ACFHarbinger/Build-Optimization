"""
Abstract base class for Hyperparameter Optimization backends.

Provides shared logic for interpreting typed search space specifications
and applying sampled parameter values to the structured Config object.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from logic.src.configs import Config

try:
    import ConfigSpace as CS
except ImportError:
    CS = None  # type: ignore[assignment,misc]

# --------------------------------------------------------------------------- #
#  Search-space spec helpers
# --------------------------------------------------------------------------- #
# Each entry in ``search_space`` is a dict with *at least* a ``type`` key:
#
#   float:       {type: float, low: 1e-5, high: 1e-3, log: true}
#   int:         {type: int,   low: 4,    high: 16,   step: 4}
#   categorical: {type: categorical, choices: [64, 128, 256]}
#
# For backwards compatibility a bare ``[low, high]`` list is auto-converted
# to ``{type: float, low: ..., high: ...}``.
# --------------------------------------------------------------------------- #

ParamSpec = Dict[str, Any]


def normalise_search_space(
    raw: Dict[str, Any],
) -> Dict[str, ParamSpec]:
    """Convert a search-space definition into canonical typed-dict form.

    Accepts both the **new** typed format and the **legacy** ``[low, high]``
    list format for backward compatibility.

    Args:
        raw: The raw ``search_space`` dict from :class:`HPOConfig`.

    Returns:
        A dict mapping dotted parameter paths to :data:`ParamSpec` dicts.
    """
    out: Dict[str, ParamSpec] = {}
    for name, spec in raw.items():
        if isinstance(spec, dict):
            # Already in canonical form
            out[name] = spec
        elif isinstance(spec, (list, tuple)) and len(spec) >= 2:
            # Legacy [low, high] shorthand
            low, high = spec[0], spec[1]
            if isinstance(low, float) or isinstance(high, float):
                out[name] = {"type": "float", "low": float(low), "high": float(high)}
            elif isinstance(low, int) and isinstance(high, int):
                out[name] = {"type": "int", "low": low, "high": high}
            else:
                # Treat as categorical choices
                out[name] = {"type": "categorical", "choices": list(spec)}
        else:
            raise ValueError(
                f"Unsupported search-space spec for '{name}': {spec!r}. Expected a typed dict or a [low, high] list."
            )
    return out


def apply_params(cfg: Config, params: Dict[str, Any]) -> Config:
    """Apply a flat dict of dotted-path parameters to a Config object.

    For example, ``{"optim.lr": 3e-4, "model.encoder.n_heads": 8}`` will
    set ``cfg.optim.lr = 3e-4`` and ``cfg.model.encoder.n_heads = 8``.

    Args:
        cfg: The root :class:`Config` instance (mutated in-place).
        params: Mapping of dotted attribute paths to values.

    Returns:
        The same *cfg* object (for convenience).
    """
    for key, value in params.items():
        parts = key.split(".")
        obj: Any = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    return cfg


# --------------------------------------------------------------------------- #
#  Abstract backend
# --------------------------------------------------------------------------- #


class BaseHPO(ABC):
    """Abstract base class for HPO backends (Optuna, DEHB, …).

    Subclasses must implement :meth:`run`, which performs the full
    optimisation loop and returns the best metric value found.

    Attributes:
        cfg: The root application configuration.
        objective_fn: Callable that trains a model for one trial and
            returns the scalar metric to **maximise**.
        search_space: Normalised search-space mapping.
    """

    def __init__(
        self,
        cfg: Config,
        objective_fn: Callable,
        search_space: Optional[Dict[str, ParamSpec]] = None,
    ):
        self.cfg = cfg
        self.objective_fn = objective_fn
        self.search_space = search_space or normalise_search_space(cfg.hpo.search_space)

    # -- abstract ---------------------------------------------------------- #

    @abstractmethod
    def run(self) -> float:
        """Execute the HPO study and return the best metric value."""
        ...

    # -- shared helpers ---------------------------------------------------- #

    @staticmethod
    def suggest_param_optuna(
        trial: Any,
        name: str,
        spec: ParamSpec,
    ) -> Any:
        """Suggest a single hyperparameter value from an Optuna trial.

        Dispatches to ``trial.suggest_float``, ``trial.suggest_int``, or
        ``trial.suggest_categorical`` based on ``spec["type"]``.

        Args:
            trial: An :class:`optuna.Trial` instance.
            name: Name used to register the suggestion.
            spec: Typed parameter specification dict.

        Returns:
            The suggested value.
        """
        ptype = spec["type"]
        if ptype == "float":
            return trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
                step=spec.get("step"),
            )
        elif ptype == "int":
            return trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
                log=spec.get("log", False),
            )
        elif ptype == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown param type '{ptype}' for '{name}'")

    @staticmethod
    def build_configspace(
        search_space: Dict[str, ParamSpec],
    ) -> Any:
        """Build a ConfigSpace object from typed specs (for DEHB).

        Args:
            search_space: Normalised search-space mapping.

        Returns:
            A :class:`ConfigSpace.ConfigurationSpace`.
        """
        if CS is None:
            raise ImportError("ConfigSpace is not installed.")

        cs = CS.ConfigurationSpace()
        for name, spec in search_space.items():
            ptype = spec["type"]
            if ptype == "float":
                hp = CS.UniformFloatHyperparameter(
                    name,
                    lower=spec["low"],
                    upper=spec["high"],
                    log=spec.get("log", False),
                )
            elif ptype == "int":
                hp = CS.UniformIntegerHyperparameter(  # type: ignore[assignment]
                    name,
                    lower=spec["low"],
                    upper=spec["high"],
                    log=spec.get("log", False),
                )
            elif ptype == "categorical":
                hp = CS.CategoricalHyperparameter(  # type: ignore[assignment]
                    name,
                    choices=spec["choices"],
                )
            else:
                raise ValueError(f"Unknown param type '{ptype}' for '{name}'")
            cs.add(hp)
        return cs
