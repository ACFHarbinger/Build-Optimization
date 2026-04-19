"""
Core handler for simulation policy hyperparameter optimization using Hydra and Optuna.
"""

import multiprocessing as mp
from typing import Any

import optuna
import torch
from logic.src import tracking as wst
from logic.src.configs import Config
from logic.src.constants import ROOT_DIR, SIM_METRICS
from logic.src.pipeline.simulations.repository import (
    load_indices,
    load_simulator_data,
    set_repository_from_path,
)
from logic.src.pipeline.simulations.simulator import sequential_simulations
from logic.src.tracking.logging.pylogger import get_pylogger
from omegaconf import OmegaConf

logger = get_pylogger(__name__)


def objective(trial: optuna.Trial, base_cfg: Config, data_size: int, lock: Any) -> float:
    """Optuna objective function for simulation policy HPO.

    Samples hyperparameters from the search space, applies them to a copy of the config,
    runs simulations, and returns the metric to maximize.

    Args:
        trial: The Optuna trial object.
        base_cfg: The base configuration to copy and mutate.
        data_size: Number of available bins for the selected area.
        lock: Multiprocessing lock for repository access.

    Returns:
        The performance metric value (higher = better).
    """
    hpo_sim = base_cfg.hpo_sim
    search_space = hpo_sim.search_space
    policy_name = hpo_sim.policy_name

    # 1. Sample from search space
    params = {}
    for name, spec in search_space.items():
        # Map parameters to the policy config path
        full_name = f"sim.full_policies.0.{policy_name}.{name}"

        p_type = spec.get("type")
        if p_type == "float":
            params[full_name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step"),
                log=spec.get("log", False),
            )
        elif p_type == "int":
            params[full_name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
                log=spec.get("log", False),
            )
        elif p_type == "categorical":
            params[full_name] = trial.suggest_categorical(name, spec["choices"])

    # 2. Clone config and apply params
    cfg_dict = OmegaConf.to_container(base_cfg, resolve=True)
    trial_cfg = OmegaConf.create(cfg_dict)

    # Ensure sim.full_policies is a list and has at least one entry for the policy
    if not trial_cfg.sim.full_policies:
        trial_cfg.sim.full_policies = [{policy_name: {}}]
    elif isinstance(trial_cfg.sim.full_policies[0], str) and trial_cfg.sim.full_policies[0] == policy_name:
        trial_cfg.sim.full_policies[0] = {policy_name: {}}
    elif isinstance(trial_cfg.sim.full_policies[0], dict) and policy_name not in trial_cfg.sim.full_policies[0]:
        # If it's a different policy, we override it for the trial
        trial_cfg.sim.full_policies[0] = {policy_name: {}}

    for key, val in params.items():
        OmegaConf.update(trial_cfg, key, val)

    # 3. Preparation for simulations
    n_samples = hpo_sim.graph.n_samples
    num_loc = hpo_sim.graph.num_loc

    if data_size != num_loc:
        focus_graph = trial_cfg.sim.graph.focus_graph
        if not focus_graph:
            wtype_suffix = f"_{hpo_sim.graph.waste_type}" if hpo_sim.graph.waste_type else ""
            focus_graph = f"graphs_{hpo_sim.graph.area}_{num_loc}V_{n_samples}N{wtype_suffix}.json"

        indices = load_indices(focus_graph, n_samples, num_loc, data_size, lock)
        if len(indices) == 1:
            indices = [indices[0]] * n_samples
    else:
        indices = [None] * n_samples  # type: ignore[assignment]

    sample_idx_ls = [list(range(n_samples))]
    device = torch.device(trial_cfg.device if hasattr(trial_cfg, "device") else "cpu")

    # 4. Run simulations
    try:
        log, _, _ = sequential_simulations(
            cfg=trial_cfg,  # type: ignore[arg-type]
            device=device,
            indices_ls=indices,
            sample_idx_ls=sample_idx_ls,
            model_weights_path="",
            lock=lock,
        )

        # 5. Extract metric
        metric_name = hpo_sim.metric
        if metric_name not in SIM_METRICS:
            idx = SIM_METRICS.index("profit") if "profit" in SIM_METRICS else 0
        else:
            idx = SIM_METRICS.index(metric_name)

        pol_name_in_log = list(log.keys())[0] if log else None
        if pol_name_in_log and log[pol_name_in_log]:
            return float(log[pol_name_in_log][idx])

        return float("-inf")

    except Exception as e:
        logger.error(f"Trial failed: {e}")
        return float("-inf")


def run_hpo_sim(cfg: Config) -> float:
    """Run Hyperparameter Optimization for simulation policies.

    Args:
        cfg: Root application configuration.

    Returns:
        Best metric value found across all trials.
    """
    sim = cfg.sim
    hpo_sim = cfg.hpo_sim

    # Initialize tracking
    experiment_name = cfg.experiment_name or f"hpo_sim_{hpo_sim.policy_name}"
    wst.init(experiment_name=experiment_name)

    # 1. Initialize Repository and resolve data size
    load_ds = getattr(cfg, "load_dataset", None)
    if load_ds is not None and set_repository_from_path(str(load_ds)):
        data_size = sim.graph.num_loc
    else:
        set_repository_from_path(str(ROOT_DIR))
        try:
            data_tmp, _ = load_simulator_data(sim.data_dir, sim.graph.num_loc, sim.graph.area, sim.graph.waste_type)
            data_size = len(data_tmp)
        except Exception:
            data_size = sim.graph.num_loc

    # 2. Setup Optuna
    manager = mp.Manager()
    lock = manager.Lock()

    study = optuna.create_study(direction="maximize")

    logger.info(f"Starting HPO for policy: {hpo_sim.policy_name}")
    logger.info(f"Method: {hpo_sim.method}, Trials: {hpo_sim.n_trials}")

    study.optimize(
        lambda trial: objective(trial, cfg, data_size, lock),
        n_trials=hpo_sim.n_trials,
        n_jobs=hpo_sim.num_workers,
    )

    logger.info(f"HPO complete. Best value ({hpo_sim.metric}): {study.best_value}")
    logger.info(f"Best config: {study.best_params}")

    # Log best results to WSTracker
    run = wst.get_active_run()
    if run is not None:
        run.log_params({f"hpo/best/{k}": v for k, v in study.best_params.items()})
        run.log_metric("hpo/best_value", study.best_value)
        run.set_tag("task", "hpo_sim")
        run.set_tag("policy", hpo_sim.policy_name)
        run.flush()

    return study.best_value
