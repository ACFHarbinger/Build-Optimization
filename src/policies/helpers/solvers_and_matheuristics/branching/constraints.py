"""
Branching constraints for VRPP Branch-and-Price.
"""

from __future__ import annotations

from typing import List, Union

from logic.src.policies.helpers.solvers_and_matheuristics.common.route import Route


class EdgeBranchingConstraint:
    """
    Directed-arc branching constraint for edge-based B&P branching.

    Fixes or forbids a single arc (u → v) in the solution.  The pricing DP
    enforces this eagerly inside the label extension loop; the heuristic
    pricer also respects it during greedy insertion.

    Attributes:
        u: Origin node of the constrained arc.
        v: Destination node of the constrained arc.
        must_use: True  → arc (u, v) MUST appear in every route that visits u
                          (x_{uv} = 1 branch).
                  False → arc (u, v) is FORBIDDEN in all routes
                          (x_{uv} = 0 branch).

    Conflict rule:
        At most one must_use arc may originate from any node, and at most one
        must_use arc may terminate at any node. Violations are detected by the
        column generation loop before the RCSPP solve and cause the B&B node
        to be marked infeasible.
    """

    def __init__(self, u: int, v: int, must_use: bool) -> None:
        """
        Initialise an edge branching constraint.

        Args:
            u: Arc origin node.
            v: Arc destination node.
            must_use: Whether the arc must (True) or must not (False) be used.
        """
        self.u = u
        self.v = v
        self.must_use = must_use

    def is_route_feasible(self, route: Route) -> bool:
        """
        Return True if *route* satisfies this edge constraint.

        Args:
            route: A Route object whose node sequence is validated.

        Returns:
            True when the constraint is not violated.
        """
        edge_present = self._edge_in_route(route.nodes)
        return edge_present if self.must_use else not edge_present

    def _edge_in_route(self, nodes: List[int]) -> bool:
        """Check whether arc (u, v) appears consecutively in the full path."""
        full_path = [0] + nodes + [0]
        return any(full_path[i] == self.u and full_path[i + 1] == self.v for i in range(len(full_path) - 1))

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        relation = "MUST_USE" if self.must_use else "FORBIDDEN"
        return f"EdgeBranchingConstraint({self.u} -> {self.v}: {relation})"


class RyanFosterBranchingConstraint:
    """
    Node-pair branching constraint for Ryan-Foster B&P branching (1981).

    Enforces co-occurrence or separation of two customer nodes across routes.
    Unlike divergence branching used in ODIMCF (Barnhart et al. 2000),
    Ryan-Foster node-pair branching is highly effective for VRPP as it
    directly modifies the RCSPP pricing subproblem without increasing
    the label state space complexity.

    Attributes:
        node_r: First node in the branching pair.
        node_s: Second node in the branching pair.
        together: True  → r and s MUST appear in the same route.
                  False → r and s MUST NOT appear in the same route.
    """

    def __init__(
        self,
        node_r: int,
        node_s: int,
        together: bool,
    ) -> None:
        """
        Initialise a Ryan-Foster branching constraint.

        Args:
            node_r: First node in the pair.
            node_s: Second node in the pair.
            together: Whether the two nodes must be co-visited (True) or
                separated (False).
        """
        self.node_r = node_r
        self.node_s = node_s
        self.together = together

    def is_route_feasible(self, route: Route) -> bool:
        """
        Return True if *route* satisfies this Ryan-Foster constraint.

        Args:
            route: A Route object to validate.

        Returns:
            True when the constraint is not violated.
        """
        r_in = self.node_r in route.node_coverage
        s_in = self.node_s in route.node_coverage

        if self.together:
            if r_in != s_in:
                return False
        else:
            if r_in and s_in:
                return False
        return True

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        relation = "TOGETHER" if self.together else "SEPARATE"
        return f"RyanFosterBranchingConstraint({self.node_r}, {self.node_s}: {relation})"


class FleetSizeBranchingConstraint:
    """Constraint on the total number of vehicles used (sum of route lambdas)."""

    def __init__(self, limit: int, is_upper: bool) -> None:
        self.limit = limit
        self.is_upper = is_upper

    def is_route_feasible(self, route: Route) -> bool:
        """Route-level feasibility is 1.0 (global constraint)."""
        return True

    def __repr__(self) -> str:
        rel = "<=" if self.is_upper else ">="
        return f"FleetSizeBranchingConstraint(Sum λ {rel} {self.limit})"


class NodeVisitationBranchingConstraint:
    """Constraint on the visitation frequency of a specific optional node."""

    def __init__(self, node: int, forced: bool) -> None:
        self.node = node
        self.forced = forced

    def is_route_feasible(self, route: Route) -> bool:
        """
        In the 'forced' branch (v_i = 1), we don't filter existing routes
        that don't visit i; the master model will enforce the visitation sum.
        In the 'forbidden' branch (v_i = 0), we filter out any route visiting i.
        """
        if not self.forced:
            return self.node not in route.node_coverage
        return True

    def __repr__(self) -> str:
        val = 1 if self.forced else 0
        return f"NodeVisitationBranchingConstraint(v_{self.node} = {val})"


# Type alias for all valid branching constraints
AnyBranchingConstraint = Union[
    EdgeBranchingConstraint,
    RyanFosterBranchingConstraint,
    FleetSizeBranchingConstraint,
    NodeVisitationBranchingConstraint,
]

# Backward-compatibility alias
BranchingConstraint = EdgeBranchingConstraint
