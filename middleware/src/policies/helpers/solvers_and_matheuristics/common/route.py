"""
Route data structure for Branch-and-Price-and-Cut VRPP.
"""

from typing import List, Optional, Set


class Route:
    """Represents a single route (column) in the master problem."""

    def __init__(
        self,
        nodes: List[int],
        cost: float,
        revenue: float,
        load: float,
        node_coverage: Set[int],
    ) -> None:
        """
        Initialise a route.

        Args:
            nodes: Sequence of customer nodes visited (excluding depot returns).
            cost: Total distance cost of the route.
            revenue: Total revenue from collected waste.
            load: Total waste collected on the route.
            node_coverage: Set of customer node IDs covered by this route.
        """
        self.nodes = nodes
        self.cost = cost
        self.revenue = revenue
        self.load = load
        self.profit = revenue - cost
        self.node_coverage = node_coverage
        self.reduced_cost: Optional[float] = None

    def __repr__(self) -> str:
        return f"Route(nodes={self.nodes}, profit={self.profit:.2f})"
