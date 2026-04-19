"""
Destroy Operators Package.

This package contains implementations of removal heuristics for ALNS, including
random, worst-case, cluster-based, Shaw (relatedness), string, route, neighbor,
historical, sector, and branch-and-bound removal.

Also includes profit-based variants for VRPP problems: random_profit_removal,
cluster_profit_removal, worst_profit_removal, shaw_profit_removal,
neighbor_profit_removal, historical_profit_removal, string_profit_removal,
sector_profit_removal, and route_profit_removal.

Example:
    >>> from logic.src.policies.helpers.operators.destroy import random_removal
    >>> routes, removed = random_removal(routes, n=5)
    >>> from logic.src.policies.helpers.operators.destroy import worst_profit_removal
    >>> routes, removed = worst_profit_removal(routes, n=5, dist_matrix=d, wastes=w, R=1.0, C=1.0)
    >>> from logic.src.policies.helpers.operators.destroy import route_profit_removal
    >>> routes, removed = route_profit_removal(routes, dist_matrix=d, wastes=w, R=1.0, C=1.0)
"""

# Unstringing/Stringing imports removed to avoid circularity with ..unstringing_stringing
from .branch_and_bound import bb_profit_removal, bb_removal
from .cluster import cluster_profit_removal, cluster_removal
from .guided import penalized_removal
from .historical import historical_profit_removal, historical_removal
from .multi_period import (
    pattern_removal,
    random_horizon_removal,
    shaw_horizon_removal,
    shift_visit_removal,
    urgency_aware_removal,
    worst_profit_horizon_removal,
)
from .neighbor import neighbor_profit_removal, neighbor_removal
from .random import random_profit_removal, random_removal
from .route import route_profit_removal, route_removal
from .sector import sector_profit_removal, sector_removal
from .shaw import shaw_profit_removal, shaw_removal
from .string import string_profit_removal, string_removal
from .worst import worst_profit_removal, worst_removal

__all__ = [
    "bb_removal",
    "bb_profit_removal",
    "random_removal",
    "worst_removal",
    "cluster_removal",
    "shaw_removal",
    "string_removal",
    "route_removal",
    "neighbor_removal",
    "historical_removal",
    "sector_removal",
    "penalized_removal",
    # Profit-based variants for VRPP
    "historical_profit_removal",
    "worst_profit_removal",
    "shaw_profit_removal",
    "route_profit_removal",
    "cluster_profit_removal",
    # Unused profit-based variants for VRPP
    "random_profit_removal",
    "neighbor_profit_removal",
    "string_profit_removal",
    "sector_profit_removal",
    # Inter-period operators
    "shift_visit_removal",
    "pattern_removal",
    "random_horizon_removal",
    "worst_profit_horizon_removal",
    "shaw_horizon_removal",
    "urgency_aware_removal",
]
