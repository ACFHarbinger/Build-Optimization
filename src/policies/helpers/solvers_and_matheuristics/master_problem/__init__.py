"""
Master Problem package for Branch-and-Price-and-Cut VRPP.

Implements the Set Partitioning Problem (SPP) formulation where each column
represents a feasible route. The master problem selects routes to maximize
total collected profit across a finite fleet.

Theoretical Basis: Barnhart et al. (1998).
"""

from __future__ import annotations

from logic.src.policies.helpers.solvers_and_matheuristics.master_problem.model import VRPPMasterProblem
from logic.src.policies.helpers.solvers_and_matheuristics.master_problem.pool import GlobalCutPool

__all__ = ["VRPPMasterProblem", "GlobalCutPool"]
