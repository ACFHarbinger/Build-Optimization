"""
Adaptive Kernel Search (AKS) matheuristic module.

This package implements the Adaptive Kernel Search algorithm as described in
the paper "Adaptive Kernel Search: A heuristic for solving Mixed Integer linear
Programs" (Guastaroba et al., 2017).

AKS is a standalone decomposition-based solver that dynamically manages the
optimization process by adjusting bucket sizes and promoting variables between
restricted sub-MIPs.
"""

from .policy_aks import AdaptiveKernelSearchPolicy

__all__ = ["AdaptiveKernelSearchPolicy"]
