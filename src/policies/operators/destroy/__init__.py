"""
Destroy Operators Package.

This package contains implementations of removal heuristics for ALNS, including
random, worst-case, cluster-based, Shaw (relatedness), and string removal.

Attributes:
    random_removal (function): Random removal heuristic.
    worst_removal (function): Worst-cost removal heuristic.
    cluster_removal (function): Cluster-based removal heuristic.
    shaw_removal (function): Shaw relatedness removal heuristic.
    string_removal (function): String removal heuristic.

Example:
    >>> from logic.src.policies.operators.destroy import random_removal
    >>> routes, removed = random_removal(routes, n=5)
"""

from .cluster import cluster_removal
from .random import random_removal
from .shaw import shaw_removal
from .string import string_removal
from .worst import worst_removal

__all__ = [
    "random_removal",
    "worst_removal",
    "cluster_removal",
    "shaw_removal",
    "string_removal",
]
