"""
Branch-and-Cut Algorithm for Vehicle Routing Problem with Profits (VRPP).

This module implements an exact branch-and-cut algorithm for VRPP based on
cutting plane methods from:
- Fischetti, M., Salazar González, J. J., & Toth, P. (1997).
  "A Branch-And-Cut Algorithm for the Symmetric Generalized Traveling Salesman Problem"
- Padberg, M., & Rinaldi, G. (1991). "A Branch-and-cut Algorithm for the Resolution
  of Large-scale Symmetric Traveling Salesman Problems"

The VRPP combines node selection (which nodes to visit) with routing optimization,
aiming to maximize profit (waste collected) minus travel cost.
"""

from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel

from .bc import BranchAndCutSolver
from .policy_bc import BranchAndCutPolicy

__all__ = ["VRPPModel", "BranchAndCutSolver", "BranchAndCutPolicy"]
