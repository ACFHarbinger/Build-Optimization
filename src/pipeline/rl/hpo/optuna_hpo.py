"""
Optuna HPO Integration.
"""

from typing import Any, Callable, Dict, Optional

import numpy as np
import optuna

from logic.src.configs import Config

try:
    from logic.src.tracking.core.run import get_active_run
except ImportError:
    get_active_run = None  # type: ignore[assignment,misc]

from .base import BaseHPO, ParamSpec


class OptunaHPO(BaseHPO):
    """
    Handles Optuna-based Hyperparameter Optimization.
    Supports TPE, Random, Grid, and Hyperband pruning.

    All parameter types are supported via the typed search-space format:
      - float (with optional log scale and step)
      - int   (with optional step and log scale)
      - categorical (list of choices)
    """

    def __init__(
        self,
        cfg: Config,
        objective_fn: Callable,
        search_space: Optional[Dict[str, ParamSpec]] = None,
    ):
        """Initialize OptunaHPO.

        Args:
            cfg: Root application configuration.
            objective_fn: Callable ``(trial, cfg) -> float`` that trains
                a model for one trial and returns the metric to maximise.
            search_space: Optional pre-normalised search space.  If *None*,
                the space is read from ``cfg.hpo.search_space``.
        """
        super().__init__(cfg, objective_fn, search_space)

    def run(self) -> float:
        """Run the optimization study.

        Returns:
            The best metric value found across all trials.
        """
        sampler = self._get_sampler()
        pruner = self._get_pruner()

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        study.optimize(
            lambda trial: self.objective_fn(trial, self.cfg),
            n_trials=self.cfg.hpo.n_trials,
            n_jobs=1,
        )

        # Log to WSTracker
        run = get_active_run() if get_active_run is not None else None
        if run is not None:
            run.log_params({f"hpo/best/{k}": v for k, v in study.best_params.items()})
            run.log_metric("hpo/best_value", study.best_value)
            run.log_metric("hpo/n_trials", len(study.trials))
            run.set_tag("hpo_backend", "optuna")
            run.set_tag("hpo_method", self.cfg.hpo.method)

        return study.best_value

    def _get_sampler(self) -> optuna.samplers.BaseSampler:
        """Build Optuna sampler based on HPO method config.

        Returns:
            An Optuna sampler instance.
        """
        seed = self.cfg.seed
        method = self.cfg.hpo.method

        if method == "random":
            return optuna.samplers.RandomSampler(seed=seed)
        elif method == "grid":
            # Build a static grid from categorical / discrete specs
            grid: Dict[str, Any] = {}
            for name, spec in self.search_space.items():
                if spec["type"] == "categorical":
                    grid[name] = spec["choices"]
                elif spec["type"] == "int":
                    step = spec.get("step", 1)
                    grid[name] = list(range(spec["low"], spec["high"] + 1, step))
                elif spec["type"] == "float":
                    # For grid search, fall back to linspace
                    grid[name] = np.linspace(spec["low"], spec["high"], 5).tolist()
            return optuna.samplers.GridSampler(grid)

        # Default: TPE
        return optuna.samplers.TPESampler(seed=seed)

    def _get_pruner(self) -> optuna.pruners.BasePruner:
        """Build Optuna pruner based on HPO method config.

        Returns:
            An Optuna pruner instance.
        """
        if self.cfg.hpo.method == "hyperband":
            return optuna.pruners.HyperbandPruner()
        return optuna.pruners.MedianPruner()
