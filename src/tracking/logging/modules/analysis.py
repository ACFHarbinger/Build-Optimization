"""
Analysis utilities for simulation logs.
"""

import json
import os
import statistics
import threading
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, cast

import constants as udef
from loguru import logger
from utils.io.files import compose_dirpath, read_json


@compose_dirpath
def load_log_dict(
    dir_paths: List[str],
    nsamples: List[int],
    show_incomplete: bool = False,
    lock: Optional[threading.Lock] = None,
) -> Dict[str, str]:
    """Load log file paths from directories keyed by graph size."""
    logs: Dict[str, str] = {}
    for path, ns in zip(dir_paths, nsamples):
        gsize = int(os.path.basename(path).split("_")[1])
        logs[f"{gsize}"] = os.path.join(path, f"log_mean_{ns}N.json")
        if show_incomplete and ns > 1:
            log_full = cast(List[Dict[str, Any]], read_json(os.path.join(path, f"log_full_{ns}N.json"), lock))
            counter: Counter[str] = Counter()
            for run in log_full:
                counter.update(run.keys())
            for key, val in dict(counter).items():
                if ns - val > 0:
                    print(f"graph {gsize} incomplete runs: - {key} - {ns - val}")
    return logs


@compose_dirpath
def output_stats(
    dir_path: str,
    nsamples: int,
    policies: List[str],
    keys: List[str],
    sort_log_func: Optional[Any] = None,
    print_output: bool = False,
    lock: Optional[threading.Lock] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute mean and std statistics for policies and write to JSON."""
    mean_filename = os.path.join(dir_path, f"log_mean_{nsamples}N.json")
    std_filename = os.path.join(dir_path, f"log_std_{nsamples}N.json")
    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return {}, {}
    try:
        if os.path.isfile(mean_filename):
            mean_dit = cast(Dict[str, Any], read_json(mean_filename, lock=None))
            std_dit = cast(Dict[str, Any], read_json(std_filename, lock=None))
        else:
            mean_dit, std_dit = {}, {}
        data = cast(List[Dict[str, Any]], read_json(os.path.join(dir_path, f"log_full_{nsamples}N.json"), lock=None))
        for pol in policies:
            tmp = [list(data[n_id][pol].values()) for n_id in range(nsamples)]
            mean_dit[pol] = {key: val for key, val in zip(keys, [*map(statistics.mean, zip(*tmp))])}
            if nsamples > 1:
                std_dit[pol] = {key: val for key, val in zip(keys, [*map(statistics.stdev, zip(*tmp))])}
            else:
                std_dit[pol] = {key: 0.0 for key in keys}

        if sort_log_func:
            mean_dit = sort_log_func(mean_dit)
            std_dit = sort_log_func(std_dit)

        if print_output:
            for pol in mean_dit:
                lg_obj = mean_dit[pol]
                lg_std_obj = std_dit[pol]
                logm = lg_obj.values() if hasattr(lg_obj, "values") else lg_obj
                logs = lg_std_obj.values() if hasattr(lg_std_obj, "values") else lg_std_obj
                print(f"{pol}:")
                for key, m, s in zip(keys, logm, logs):
                    print(f"- {key}: {m:.2f} +- {s:.4f}")
        with open(mean_filename, "w") as fp:
            json.dump(mean_dit, fp, indent=True)
        with open(std_filename, "w") as fp:
            json.dump(std_dit, fp, indent=True)
    finally:
        if lock is not None:
            lock.release()
    return mean_dit, std_dit


@compose_dirpath
def runs_per_policy(
    dir_paths: List[str],
    nsamples: List[int],
    policies: List[str],
    print_output: bool = False,
    lock: Optional[threading.Lock] = None,
) -> List[Dict[str, List[int]]]:
    """Count runs per policy from full log files."""
    runs_ls = []
    for path, ns in zip(dir_paths, nsamples):
        dit: Dict[str, List[int]] = {pol: [] for pol in policies}
        data = cast(List[Dict[str, Any]], read_json(os.path.join(path, f"log_full_{ns}N.json"), lock))
        for id, run_data in enumerate(data):
            for key in dit:
                if key in run_data:
                    dit[key].append(id)
        runs_ls.append(dit)
        if print_output:
            gsize = int(os.path.basename(path).rsplit("_", 1)[1])
            print(f"graph {gsize} #runs per policy:")
            for key, val in dit.items():
                print(f"- {key}: {len(val)} samples: {val}")
    return runs_ls


def final_simulation_summary(log: Dict[str, Any], policy: str, n_samples: int) -> None:
    """Log a final summary of simulation statistics for a policy."""
    if policy not in log:
        logger.warning(f"Policy {policy} not found in log for summary.")
        return
    stats = log[policy]
    logger.info(f"=== Simulation Summary: {policy} ({n_samples} samples) ===")
    for metric in ["overflows", "kg", "km", "kg/km", "profit"]:
        if metric in stats:
            val = stats[metric]
            logger.info(f" - {metric:10}: {val:>10.2f}")
    logger.info("================================================")
