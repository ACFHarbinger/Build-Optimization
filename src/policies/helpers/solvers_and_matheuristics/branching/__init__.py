"""
Branching package for VRPP Branch-and-Price-and-Cut.

Provides several constraint types and the supporting B&B tree infrastructure:

EdgeBranchingConstraint
    Operates on directed arcs (u → v).  Integrates cleanly with the DP label
    extension step: forbidden / required arcs are enforced in O(1) per
    extension without any post-hoc filtering.
    Reference: Barnhart et al. (1998), Section 4.

RyanFosterBranchingConstraint
    Operates on *node pairs* (r, s).  Requires routes to either always contain
    both nodes in the same route (`together=True`) or never contain them
    together (`together=False`).
    Reference: Ryan & Foster (1981), Proposition 1.

FleetSizeBranchingConstraint
    Operates on the total number of vehicles used (sum of lambda variables).
    Enforces a floor or ceiling on the total fleet size.

NodeVisitationBranchingConstraint
    Operates on a single node i. Enforces v_i = 0 (forbidden) or v_i = 1 (forced).

MultiEdgePartitionBranching
    Spatial fleet-partitioning heuristic. Uses polar mapping geometry to
    forbid sets of arcs across different branches, yielding stronger
    polyhedral divergence for anonymous fleets.
    Reference: Barnhart et al. (1998, 2000).

Theoretical Framework:
----------------------
Ryan-Foster branching is utilized for VRPP because it appropriately modifies
the Resource-Constrained Shortest Path Problem (RCSPP) used for pricing by
enforcing or forbidding node pairs. In a Set Partitioning Problem (SPP),
node-pair branching is mathematically exact and avoids the "symmetry" issues
inherent in arc-based branching for problems with multiple identical vehicles.
While Barnhart et al. (2000) avoids this for ODIMCF to maintain simple
shortest-path pricing, the RCSPP pricing in VRPP is already weakly NP-hard,
and node-pair constraints are easily integrated into the DP label extension.
"""

from __future__ import annotations

from .constraints import (
    AnyBranchingConstraint,
    BranchingConstraint,
    EdgeBranchingConstraint,
    FleetSizeBranchingConstraint,
    NodeVisitationBranchingConstraint,
    RyanFosterBranchingConstraint,
)
from .strategies import (
    EdgeBranching,
    FleetSizeBranching,
    MultiEdgePartitionBranching,
    NodeVisitationBranching,
    RyanFosterBranching,
)
from .tree import BranchAndBoundTree

__all__ = [
    "EdgeBranchingConstraint",
    "RyanFosterBranchingConstraint",
    "FleetSizeBranchingConstraint",
    "NodeVisitationBranchingConstraint",
    "AnyBranchingConstraint",
    "BranchingConstraint",
    "EdgeBranching",
    "RyanFosterBranching",
    "FleetSizeBranching",
    "NodeVisitationBranching",
    "MultiEdgePartitionBranching",
    "BranchAndBoundTree",
]
