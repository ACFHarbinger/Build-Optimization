"""
Hyper-Parameter Optimization (HPO) constants.

This module defines configuration keys for HPO experiments.
Used by:
- logic/src/pipeline/rl/hpo/ (Optuna, DEHB, Ray Tune)
- logic/src/configs/hpo.py (HPO config dataclass validation)
- logic/src/cli/ts_parser.py (CLI argument parsing for HPO commands)

HPO Methods Supported
---------------------
- **Optuna**: Bayesian optimization with TPE, NSGA-II, CMA-ES samplers
- **DEHB**: Differential Evolution Hyperband (efficient multi-fidelity)
- **Ray Tune**: Distributed HPO with ASHA, HyperBand, Bayesian optimization
- **Grid Search**: Exhaustive search over discrete parameter grids

Key Naming Conventions
----------------------
- Optuna-specific: n_trials, n_startup_trials, n_warmup_steps, timeout
- DEHB-specific: eta, max_tres, reduction_factor, fevals
- Evolutionary (NSGA-II): indpb, tournsize, cxpb, mutpb, n_pop, n_gen
- Ray Tune-specific: num_samples, max_failures, max_conc
- Shared: hpo_method, metric, cpu_cores, verbose, train_best

Usage Context
-------------
These keys validate config files and CLI arguments. Missing keys trigger
default value fallbacks in logic/src/configs/hpo.py.
"""

from typing import Tuple

# Hyper-Parameter Optimization configuration keys
# Tuple ensures immutability (config keys should never change at runtime)
HOP_KEYS: Tuple[str, ...] = (
    # Core HPO settings (all methods)
    "hpo_method",  # Algorithm: "optuna", "dehb", "ray", "grid"
    "hpo_range",  # Parameter search space definition (dict or list)
    "hpo_epochs",  # Training epochs per trial (budget per evaluation)
    "metric",  # Objective to optimize: "cost", "profit", "kg/km"
    "cpu_cores",  # Parallel workers for trial execution
    "verbose",  # Logging verbosity (0=silent, 1=progress, 2=debug)
    "train_best",  # Whether to retrain best config on full data after HPO
    "local_mode",  # Run Ray Tune locally (vs distributed cluster)
    # Optuna-specific parameters
    "n_trials",  # Total optimization trials (budget)
    "timeout",  # Max HPO time (seconds), trials stop after this
    "n_startup_trials",  # Random trials before Bayesian optimization starts
    "n_warmup_steps",  # Steps before pruner starts early-stopping trials
    "interval_steps",  # Pruning check frequency (every N steps)
    # DEHB-specific parameters
    "eta",  # Successive halving reduction factor (default: 3)
    "max_tres",  # Maximum resources per trial (e.g., max epochs)
    "reduction_factor",  # Budget reduction between rungs (HyperBand)
    "fevals",  # Function evaluations budget (total trials × epochs)
    # Evolutionary algorithm parameters (NSGA-II, NSGA-III)
    "indpb",  # Independent mutation probability per gene
    "tournsize",  # Tournament selection size (higher = more elitism)
    "cxpb",  # Crossover probability (0.0-1.0)
    "mutpb",  # Mutation probability (0.0-1.0)
    "n_pop",  # Population size (number of solutions per generation)
    "n_gen",  # Number of generations (evolutionary iterations)
    # Ray Tune-specific parameters
    "num_samples",  # Total trials to run (Ray Tune terminology)
    "max_failures",  # Trial failures before stopping (fault tolerance)
    "max_conc",  # Max concurrent trials (parallel execution limit)
    # Grid search parameters
    "grid",  # Grid definition: dict of parameter lists to combine
)
