"""
Pricing package for Branch-and-Price-and-Cut.

Solves the Resource-Constrained Shortest Path Problem (RCSPP) to identify
profitable columns (routes) for the BPC Master Problem.

Theoretical Deviation Note:
---------------------------
Unlike the pure ODIMCF formulation in Barnhart, Hane, and Vance (2000) which
manages state-space explosion strictly via branching, this implementation intentionally
incorporates Subset-Row Inequalities (SRIs) and accommodates Ryan-Foster branching.
These algorithmic choices fundamentally expand the DP state space by requiring auxiliary
resource dimensions (e.g., tracking SRI parity constraints). To mitigate this explosion
and maintain tractability, we rely heavily on the ng-route relaxation framework,
balancing bounding strength with computational viability for the VRPP.
"""

from __future__ import annotations

from logic.src.policies.helpers.solvers_and_matheuristics.pricing.labels import Label
from logic.src.policies.helpers.solvers_and_matheuristics.pricing.solver import RCSPPSolver

__all__ = ["Label", "RCSPPSolver"]
