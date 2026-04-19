"""Ray Tune HPO backend for WSmart-Route.

Provides :class:`RayTuneHPO`, a :class:`BaseHPO` subclass that uses
`Ray Tune <https://docs.ray.io/en/latest/tune/index.html>`_ to run
distributed hyperparameter sweeps with advanced schedulers:

* **ASHA** (``"asha"``) — Asynchronous Successive Halving Algorithm.
  Best default: early-stops bad trials aggressively, very GPU-efficient.
* **PBT** (``"pbt"``) — Population Based Training.
  Evolves a *population* of configurations mid-training via exploit+explore.
* **BOHB** (``"bohb"``) — Bayesian Optimisation with HyperBand.
  Uses BOHB sampler paired with HyperBand-for-BOHB scheduler.

Search space
------------
The same typed ``ParamSpec`` format used by Optuna and DEHB is automatically
converted to Ray Tune's ``param_space`` dict:

.. code-block:: yaml

    hpo:
      method: asha
      search_space:
        optim.lr:   {type: float, low: 1e-5, high: 1e-3, log: true}
        rl.entropy_weight: {type: float, low: 0.0,  high: 0.1}
        model.encoder.n_layers: {type: int, low: 2, high: 6}

MLflow integration
------------------
When ``ray_tune_mlflow_enabled=True`` in :class:`TrackingConfig`, every
trial automatically logs to MLflow via ``MLflowLoggerCallback``.

Typical usage
-------------
This class is instantiated automatically by ``run_hpo()`` in
``logic/src/pipeline/features/train/hpo.py`` when ``cfg.hpo.method`` is
one of ``"asha"``, ``"pbt"``, or ``"bohb"``.
"""

from __future__ import annotations

import contextlib
import copy
from typing import Any, Callable, Dict, Optional

from logic.src.configs import Config

try:
    import ray
    import ray.train as ray_train
    from ray import tune
    from ray import tune as rt
    from ray.tune.logger.mlflow import MLflowLoggerCallback
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
    from ray.tune.search.bohb import TuneBOHB

    from logic.src.tracking.core.run import get_active_run
except ImportError:
    ray = None  # type: ignore[assignment,misc]
    ray_train = None  # type: ignore[assignment,misc]
    tune = None  # type: ignore[assignment,misc]
    rt = None  # type: ignore[assignment,misc]
    MLflowLoggerCallback = None  # type: ignore[assignment,misc]
    ASHAScheduler = None  # type: ignore[assignment,misc]
    PopulationBasedTraining = None  # type: ignore[assignment,misc]
    HyperBandForBOHB = None  # type: ignore[assignment,misc]
    TuneBOHB = None  # type: ignore[assignment,misc]
    get_active_run = None  # type: ignore[assignment,misc]

from .base import BaseHPO, ParamSpec, apply_params


class RayTuneHPO(BaseHPO):
    """Ray Tune HPO backend with ASHA / PBT / BOHB schedulers.

    Each trial runs in a Ray worker process, builds its own model and
    ``WSTrainer``, and reports ``val_reward`` back to the Tune scheduler
    after every epoch via ``ray.train.report()``.

    Args:
        cfg: Root application configuration.
        objective_fn: Callable ``(trial_cfg: Config) -> float`` that
            trains a model for one trial and returns the validation
            reward to **maximise**.  The function must be importable in
            worker processes (i.e. it should be a top-level function, not
            a lambda or local closure).
        search_space: Optional pre-normalised search space dict.  When
            ``None``, ``cfg.hpo.search_space`` is used.
        scheduler: Which scheduler to use.  One of ``"asha"`` (default),
            ``"pbt"``, or ``"bohb"``.
        mlflow_tracking_uri: MLflow server URI forwarded to
            ``MLflowLoggerCallback``.  ``None`` disables per-trial
            MLflow logging.
        mlflow_experiment_name: MLflow experiment name for trial runs.
    """

    def __init__(
        self,
        cfg: Config,
        objective_fn: Callable,
        search_space: Optional[Dict[str, ParamSpec]] = None,
        scheduler: str = "asha",
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = None,
    ) -> None:
        super().__init__(cfg, objective_fn, search_space)
        self._scheduler_name = scheduler
        self._mlflow_tracking_uri = mlflow_tracking_uri
        self._mlflow_experiment_name = mlflow_experiment_name

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> float:
        """Execute the Ray Tune study and return the best metric value.

        Returns:
            Best ``val_reward`` found across all trials, or ``0.0`` if
            no successful trial was recorded.
        """
        if ray is None:
            raise ImportError("ray is not installed")

        if not ray.is_initialized():
            ray.init(
                local_mode=self.cfg.hpo.local_mode,
                ignore_reinit_error=True,
                include_dashboard=False,
            )

        ray_space = self._build_ray_search_space()
        scheduler = self._build_scheduler()
        search_alg = self._build_search_alg()
        run_callbacks = self._build_run_callbacks()

        # Capture cfg snapshot for serialisation into worker closures
        cfg_snapshot = copy.deepcopy(self.cfg)
        objective_fn = self.objective_fn

        def trainable(trial_config: Dict[str, Any]) -> None:
            """Per-trial trainable executed inside a Ray worker."""
            trial_cfg = copy.deepcopy(cfg_snapshot)
            apply_params(trial_cfg, trial_config)

            # Run the objective; it is expected to return a float
            val_reward = float("-inf")
            with contextlib.suppress(Exception):
                val_reward = float(objective_fn(trial_cfg))

            ray_train.report({"val_reward": val_reward})

        gpu_per_trial = 1.0 if self.cfg.device == "cuda" else 0.0
        cpu_per_trial = max(1, self.cfg.hpo.cpu_cores)

        tuner = tune.Tuner(
            tune.with_resources(
                trainable,
                resources={"cpu": cpu_per_trial, "gpu": gpu_per_trial},
            ),
            param_space=ray_space,
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                search_alg=search_alg,
                num_samples=self.cfg.hpo.num_samples,
                metric="val_reward",
                mode="max",
                max_concurrent_trials=self.cfg.hpo.max_conc,
            ),
            run_config=ray.train.RunConfig(
                storage_path=getattr(self.cfg.tracking, "ray_tune_storage_path", "ray_results"),
                failure_config=ray.train.FailureConfig(
                    max_failures=self.cfg.hpo.max_failures,
                ),
                callbacks=run_callbacks,
            ),
        )

        results = tuner.fit()

        best = None
        with contextlib.suppress(Exception):
            best = results.get_best_result(metric="val_reward", mode="max")

        best_val = 0.0
        if best is not None and best.metrics:
            best_val = float(best.metrics.get("val_reward", 0.0))

        # Log to WSTracker
        run = get_active_run() if get_active_run is not None else None
        if run is not None:
            run.log_metric("hpo/best_val_reward", best_val)
            run.log_metric("hpo/num_samples", self.cfg.hpo.num_samples)
            run.set_tag("hpo_backend", "ray_tune")
            run.set_tag("hpo_scheduler", self._scheduler_name)
            if best is not None and best.config:
                run.log_params({f"hpo/best/{k}": v for k, v in best.config.items()})

        return best_val

    # ------------------------------------------------------------------
    # Search space conversion
    # ------------------------------------------------------------------

    def _build_ray_search_space(self) -> Dict[str, Any]:
        """Convert typed ParamSpec dict to Ray Tune ``param_space``."""
        space: Dict[str, Any] = {}
        for name, spec in self.search_space.items():
            ptype = spec["type"]
            if ptype == "float":
                if spec.get("log", False):
                    space[name] = rt.loguniform(spec["low"], spec["high"])
                elif spec.get("step") is not None:
                    # Quantised uniform — approximate with quniform
                    space[name] = rt.quniform(spec["low"], spec["high"], spec["step"])
                else:
                    space[name] = rt.uniform(spec["low"], spec["high"])
            elif ptype == "int":
                space[name] = rt.randint(spec["low"], spec["high"] + 1)
            elif ptype == "categorical":
                space[name] = rt.choice(spec["choices"])
            else:
                raise ValueError(f"Unknown param type '{ptype}' for '{name}'")
        return space

    # ------------------------------------------------------------------
    # Scheduler & search algorithm factories
    # ------------------------------------------------------------------

    def _build_scheduler(self) -> Any:
        """Instantiate the Ray Tune trial scheduler."""
        name = self._scheduler_name
        max_t = self.cfg.hpo.n_epochs_per_trial
        grace = max(1, max_t // self.cfg.hpo.reduction_factor)

        if name == "asha":
            return ASHAScheduler(
                max_t=max_t,
                grace_period=grace,
                reduction_factor=self.cfg.hpo.reduction_factor,
            )

        if name == "pbt":
            # Build perturbation intervals from the search space
            hyperparam_mutations: Dict[str, Any] = {}
            for param_name, spec in self.search_space.items():
                if spec["type"] == "categorical":
                    hyperparam_mutations[param_name] = spec["choices"]
                else:
                    hyperparam_mutations[param_name] = [spec["low"], spec["high"]]

            return PopulationBasedTraining(
                time_attr="training_iteration",
                perturbation_interval=self.cfg.hpo.interval_steps,
                hyperparam_mutations=hyperparam_mutations,
            )

        if name == "bohb":
            return HyperBandForBOHB(
                time_attr="training_iteration",
                max_t=max_t,
                reduction_factor=self.cfg.hpo.reduction_factor,
            )

        # Fallback: ASHA

        return ASHAScheduler(max_t=max_t, grace_period=grace)

    def _build_search_alg(self) -> Optional[Any]:
        """Return a search algorithm compatible with the chosen scheduler."""
        if self._scheduler_name == "bohb":
            # BOHB requires the TuneBOHB search algorithm
            with contextlib.suppress(ImportError):
                return TuneBOHB(seed=self.cfg.seed)
        return None

    # ------------------------------------------------------------------
    # MLflow per-trial callback
    # ------------------------------------------------------------------

    def _build_run_callbacks(self) -> list:
        """Build Ray Tune run-level callbacks."""
        callbacks: list = []

        mlflow_uri = self._mlflow_tracking_uri
        mlflow_exp = self._mlflow_experiment_name

        if not mlflow_uri:
            return callbacks

        with contextlib.suppress(ImportError, Exception):
            callbacks.append(
                MLflowLoggerCallback(
                    tracking_uri=mlflow_uri,
                    experiment_name=mlflow_exp or "wsmart-route-hpo",
                    save_artifact=True,
                )
            )
        return callbacks
