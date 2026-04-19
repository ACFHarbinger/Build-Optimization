"""
Dynamic programming labels for RCSPP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Set, Tuple


@dataclass(order=True)
class Label:
    """
    Label for dynamic programming state in RCSPP / ng-RCSPP.

    Represents a partial path from the depot to the current node with
    accumulated resources.  Labels are ordered by reduced cost for efficient
    dominance checking (higher is better for the maximisation objective).
    """

    # Primary sort key — higher reduced cost is preferred.
    reduced_cost: float = field(compare=True)

    # State fields — excluded from the dataclass ordering.
    node: int = field(compare=False)
    load: float = field(compare=False)

    # visited: complete set of customer nodes on the partial path.
    visited: Set[int] = field(default_factory=set, compare=False)

    # ng_memory: compact relaxed state for ng-route dominance / feasibility.
    ng_memory: Set[int] = field(default_factory=set, compare=False)

    # rf_unmatched: nodes from a 'together' pair visited without their partner.
    # Used for enforcing Ryan-Foster branching constraints during pricing.
    rf_unmatched: FrozenSet[int] = field(default_factory=frozenset, compare=False)

    parent: Optional["Label"] = field(default=None, compare=False, repr=False)

    # Subset-Row Inequalities (SRI) state:
    # A tuple where each entry corresponds to an active SRI subset S.
    # State values:
    #   0: No nodes in S visited yet.
    #   1: One node in S visited (potential penalty on next visit).
    #   2: Two nodes in S visited (dual penalty applied, resets on 3rd visit if allowed).
    # This state enables the exact calculation of ⌊ 1/2 * Σ a_{ik} ⌋ dual penalties.
    sri_state: Tuple[int, ...] = field(default_factory=tuple, compare=False)

    def dominates(
        self,
        other: "Label",
        use_ng: bool = False,
        epsilon: float = 1e-6,
        sri_dual_values: Optional[List[float]] = None,
    ) -> bool:
        """
        Check if this label dominates another label.

        Dominance criteria:
        1. Same node
        2. Lower or equal load
        3. Better or equal reduced cost (accounting for SRI potential)
        4. Smaller or equal subset of visited/ng-memory nodes
        5. Subset-row inequality state compatibility
        6. Ryan-Foster conflict set compatibility
        """
        if self.node != other.node:
            return False
        if self.load > other.load + epsilon:
            return False
        if not self.rf_unmatched.issubset(other.rf_unmatched):
            return False

        # Exact ESPPRC requirement for SRI states
        if self.sri_state != other.sri_state:
            return False

        total_potential_penalty = 0.0
        if sri_dual_values is not None:
            for s, o, dual in zip(self.sri_state, other.sri_state, sri_dual_values, strict=False):
                if s == 1 and o in (0, 2):
                    total_potential_penalty += dual
        else:
            if any(s > o for s, o in zip(self.sri_state, other.sri_state, strict=False)):
                return False

        if self.reduced_cost - total_potential_penalty < other.reduced_cost - epsilon:
            return False

        if use_ng:
            return self.ng_memory.issubset(other.ng_memory)
        else:
            return self.visited.issubset(other.visited)

    def is_feasible(self, capacity: float) -> bool:
        """Check if the label satisfies resource constraints."""
        return self.load <= capacity

    def reconstruct_path(self) -> List[int]:
        """Backtrack from the current label to the start node to recover the full route."""
        if self.parent is None:
            return [self.node]
        return self.parent.reconstruct_path() + [self.node]
