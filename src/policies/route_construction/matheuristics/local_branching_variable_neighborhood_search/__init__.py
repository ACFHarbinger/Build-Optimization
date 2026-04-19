"""
Local Branching with Variable Neighborhood Search (LB-VNS) matheuristic module.

This package implements the LB-VNS algorithm, a hybrid optimization approach
that leverages the exploration capabilities of Variable Neighborhood Search (VNS)
to escape local optima, while using Local Branching (LB) for rigorous
intensification within the mathematical programming search space.

Based on:
    Hansen, P., Mladenović, N., & Urošević, D. (2006).
    "Variable neighborhood search and local branching".
    Computers & Operations Research.
"""

from .policy_lb_vns import LocalBranchingVNSPolicy

__all__ = ["LocalBranchingVNSPolicy"]
