"""
Particle Swarm Optimization (PSO) with velocity momentum.

**TRUE PSO** replacing the mathematically equivalent but slower SCA.
"""

from .params import PSOParams
from .policy_pso import PSOPolicyAdapter
from .solver import PSOSolver

__all__ = ["PSOParams", "PSOPolicyAdapter", "PSOSolver"]
