"""
Repair Operators Package.

This package contains operator implementations for repairing (re-inserting nodes into)
partial solutions. Includes greedy, regret-based, savings-based, deep, and
branch-and-bound heuristics.

Example:
    >>> from logic.src.policies.helpers.operators.repair import greedy_insertion
    >>> routes = greedy_insertion(routes, removed, dist_matrix, wastes, capacity)
    >>> from logic.src.policies.helpers.operators.repair import bb_insertion
    >>> routes = bb_insertion(routes, removed, dist_matrix, wastes, capacity)
"""

from .branch_and_bound import bb_insertion, bb_profit_insertion
from .deep import deep_insertion, deep_profit_insertion
from .farthest import farthest_insertion, farthest_profit_insertion
from .forward_looking import forward_looking_insertion
from .geni import geni_insertion, geni_profit_insertion
from .greedy import greedy_insertion, greedy_profit_insertion
from .greedy_blink import greedy_insertion_with_blinks, greedy_profit_insertion_with_blinks
from .multi_period import greedy_horizon_insertion, regret_k_temporal_insertion, stochastic_aware_insertion
from .nearest import nearest_insertion, nearest_profit_insertion
from .regret import (
    regret_2_insertion,
    regret_2_profit_insertion,
    regret_3_insertion,
    regret_3_profit_insertion,
    regret_4_insertion,
    regret_4_profit_insertion,
    regret_k_insertion,
    regret_k_profit_insertion,
)
from .savings import savings_insertion, savings_profit_insertion

__all__ = [
    "bb_insertion",
    "bb_profit_insertion",
    "greedy_insertion",
    "greedy_profit_insertion",
    "farthest_insertion",
    "farthest_profit_insertion",
    "geni_insertion",
    "geni_profit_insertion",
    "regret_2_insertion",
    "regret_2_profit_insertion",
    "regret_3_insertion",
    "regret_3_profit_insertion",
    "regret_4_insertion",
    "regret_4_profit_insertion",
    "regret_k_insertion",
    "regret_k_profit_insertion",
    "greedy_insertion_with_blinks",
    "greedy_profit_insertion_with_blinks",
    "nearest_insertion",
    "nearest_profit_insertion",
    "savings_insertion",
    "savings_profit_insertion",
    "deep_insertion",
    "deep_profit_insertion",
    # Inter-period / forward-looking
    "forward_looking_insertion",
    "greedy_horizon_insertion",
    "regret_k_temporal_insertion",
    "stochastic_aware_insertion",
]
