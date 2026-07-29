"""
Statistical functions and constants.

This module provides a registry of statistical operations used for:
- Simulation result analysis (parsing logs, computing aggregates)
- Hyperparameter optimization (computing objective function values)
- GUI data visualization (chart statistics, summary tables)

STATS_FUNCTION_MAP Usage Context
--------------------------------
Maps string identifiers to statistical functions from Python's statistics module
and built-ins. Used by:

1. **Simulation analyzer**: logic/src/pipeline/simulations/actions/logging.py
   - Computes daily/overall statistics for kg, km, overflows, etc.
   - Example: stats["mean"](daily_costs) → average daily cost

2. **GUI chart workers**: gui/src/helpers/chart_worker.py
   - Generates summary statistics for interactive plots
   - Example: stats["median"](route_lengths) → typical route length

3. **HPO objective functions**: logic/src/pipeline/rl/hpo/
   - Reduces multi-run results to single scalar value
   - Example: stats["min"](validation_losses) → best validation loss

4. **Notebook analysis**: notebooks/*.ipynb
   - Quick statistical summaries of experimental results
   - Example: stats["quant"](profits, n=4) → profit quartiles

All functions accept iterables (list, tuple, numpy array) and return scalars
except "quant" which returns a list of quantile values.
"""

import statistics
from typing import Any, Callable, Dict

# Statistical function registry mapping string keys to callables.
# All functions accept iterable inputs; output types noted in comments.
STATS_FUNCTION_MAP: Dict[str, Callable[..., Any]] = {
    # Central tendency measures (return float)
    "mean": statistics.mean,  # Arithmetic average. Use for normally distributed data.
    "median": statistics.median,  # Middle value. Robust to outliers (e.g., route costs with anomalies).
    "mode": statistics.mode,  # Most common value. Use for categorical/discrete data (e.g., most frequent tour length).
    # Dispersion measures (return float)
    "stdev": statistics.stdev,  # Sample standard deviation (√variance). Measures spread around mean.
    "var": statistics.variance,  # Sample variance (σ²). Use when comparing variability across distributions.
    # Distribution analysis (returns List[float])
    "quant": statistics.quantiles,  # Quantile values. Default n=4 (quartiles). Use: stats["quant"](data, n=10) for deciles.
    # Aggregation functions (return int or float)
    "size": len,  # Count of elements. Use for sample size reporting.
    "sum": sum,  # Total of all values. Use for cumulative metrics (total kg collected, total cost).
    "min": min,  # Minimum value. Use for best-case performance (lowest cost, shortest route).
    "max": max,  # Maximum value. Use for worst-case performance (peak overflow, longest route).
}
