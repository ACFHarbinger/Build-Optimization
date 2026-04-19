"""
State-space search tree node for Branch-and-Bound.

Each node in the tree represents a subproblem where a subset of binary decision
variables (edges and node visits) has been fixed to specific integer values.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from logic.src.policies.helpers.solvers_and_matheuristics.branching.constraints import AnyBranchingConstraint
    from logic.src.policies.helpers.solvers_and_matheuristics.common.route import Route


@dataclass(order=True)
class Node:
    """Represents a discrete node in the Branch-and-Bound exploration tree.

    Nodes are prioritized in the search queue based on their lower bound (LB)
    value, implementing a best-bound-first search strategy to minimize tree size.

    Attributes:
        bound (float): The LP relaxation objective value at this node.
            Serves as an upper bound on any integer solution reachable from
            this node (for maximization problems).
        fixed_x (Dict[Tuple[int, int], int]): A dictionary mapping edge tuples (i, j)
            to their fixed binary values (0 or 1).
        fixed_y (Dict[int, int]): A dictionary mapping customer IDs to their
            fixed binary visit statuses (0 or 1).
        depth (int): The distance of the node from the root of the search tree.
    """

    bound: float = field(compare=True)  # LP relaxation value (upper bound for maximization)
    fixed_x: Dict[Tuple[int, int], int] = field(compare=False, default_factory=dict)
    fixed_y: Dict[int, int] = field(compare=False, default_factory=dict)
    depth: int = field(compare=False, default=0)


class BranchNode:
    """
    A single node in the Branch-and-Price-and-Cut search tree.

    Holds the branch-specific constraints and the resulting LP relaxation
    bound discovered during column generation at this node.

    Attributes:
        constraints: List of EdgeBranchingConstraint or
            RyanFosterBranchingConstraint objects active at this node.
        lp_bound: Objective value of the RMP linear relaxation (z_LP).
            Serves as a dual bound for all child nodes.
        lp_basis: Tuple of (vbasis, cbasis) Gurobi attributes used to warm-start
            the master problem solve from the parent's solution.
        routes: Optional subset of columns (Route objects) maintained
            locally to accelerate RMP convergence.
    """

    def __init__(
        self,
        constraints: Optional[List["AnyBranchingConstraint"]] = None,
        parent: Optional["BranchNode"] = None,
        depth: int = 0,
        lp_bound_hint: Optional[float] = None,
        branching_rule: str = "none",
    ) -> None:
        """
        Initialise a branch node.

        Args:
            constraints: Constraints added *at this node only* (not inherited).
            parent: Parent node (None for root).
            depth: Tree depth (root = 0).
            lp_bound_hint: Optional initial LP bound (inherited from parent).
            branching_rule: Name of the rule used to create this node's branch.
        """
        self.constraints: List["AnyBranchingConstraint"] = constraints or []
        self.parent: Optional["BranchNode"] = parent
        self.depth: int = depth
        self.branching_rule = branching_rule

        # State tracking
        self.lp_bound: Optional[float] = lp_bound_hint
        self.ip_solution: Optional[float] = None
        self.is_integer: bool = False
        self.is_infeasible: bool = False
        self.is_explored: bool = False
        self.is_pruned: bool = False

        # Solution data
        self.route_values: Optional[Dict[int, float]] = None
        self.routes: Optional[List["Route"]] = None
        self.lp_basis: Optional[Any] = None

        self.children: List["BranchNode"] = []

    def get_all_constraints(self) -> List["AnyBranchingConstraint"]:
        """
        Return all active constraints from the root down to this node.

        Returns:
            Flat list in root-to-leaf order.
        """
        constraints: List["AnyBranchingConstraint"] = []
        node: Optional["BranchNode"] = self
        while node is not None:
            # Reverse node local constraints to keep consistent order if multiple exist
            constraints.extend(node.constraints[::-1])
            node = node.parent
        return constraints[::-1]

    def is_route_feasible(self, route: "Route") -> bool:
        """
        Return True if *route* satisfies every inherited constraint.

        Args:
            route: Route to validate.
        """
        return all(c.is_route_feasible(route) for c in self.get_all_constraints())

    def __lt__(self, other: "BranchNode") -> bool:
        """Priority for best-first search (higher bound is better for maximization)."""
        b1 = self.lp_bound if self.lp_bound is not None else -1e10
        b2 = other.lp_bound if other.lp_bound is not None else -1e10
        return b1 > b2

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        bound_str = f"{self.lp_bound:.2f}" if self.lp_bound is not None else "None"
        return f"BranchNode(depth={self.depth}, bound={bound_str}, constraints={len(self.constraints)})"
