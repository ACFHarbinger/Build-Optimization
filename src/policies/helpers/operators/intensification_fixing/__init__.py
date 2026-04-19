"""
Intensification operators: Held-Karp DP, Fix-and-Optimize Sub-MIP, and Set-Partitioning Polish.
"""

from .dp_route_reopt import dp_route_reopt, dp_route_reopt_profit
from .fix_and_optimize import fix_and_optimize, fix_and_optimize_profit
from .set_partitioning_polish import set_partitioning_polish, set_partitioning_polish_profit

INTENSIFICATION_OPERATORS = {
    "DP_REOPT": dp_route_reopt,
    "DP_REOPT_PROFIT": dp_route_reopt_profit,
    "FIX_OPT": fix_and_optimize,
    "FIX_OPT_PROFIT": fix_and_optimize_profit,
    "SP_POLISH": set_partitioning_polish,
    "SP_POLISH_PROFIT": set_partitioning_polish_profit,
}

INTENSIFICATION_NAMES = list(INTENSIFICATION_OPERATORS.keys())

__all__ = [
    "INTENSIFICATION_NAMES",
    "INTENSIFICATION_OPERATORS",
    "dp_route_reopt",
    "dp_route_reopt_profit",
    "fix_and_optimize",
    "fix_and_optimize_profit",
    "set_partitioning_polish",
    "set_partitioning_polish_profit",
]
