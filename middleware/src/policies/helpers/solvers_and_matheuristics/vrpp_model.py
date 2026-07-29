r"""
VRPP Model Formulation for Branch-and-Price-and-Cut (BPC).

Implements the polyhedral structure and physical constraints for the Vehicle
Routing Problem with Profits.

Conceptual Link to BPC:
-----------------------
The Natural Edge Formulation $(x, y)$ provides the theoretical basis for our
branching and cutting decisions. However, the BPC engine operates on an
equivalent Set Partitioning formulation $(\lambda)$ for computational efficiency.
This model bridges the two by:
1. Providing graph topology for RCSPP pricing.
2. Formulating base constraints (degree, capacity) that are strengthened by
   dynamic cut separation.
3. Defining the node selection and profit logic.

References:
-----------
- Fischetti, M., Salazar-González, J. J., & Toth, P. (1997). "A branch-and-cut
  algorithm for the symmetric generalized traveling salesman problem."
  Operations Research, 45(3), 378-394.
"""

from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np


class VRPPModel:
    """
    Mathematical model for the Vehicle Routing Problem with Profits (VRPP).

    This solver adapts the Symmetric Generalized TSP (GTSP) infrastructure of
    Fischetti et al. (1997) to the Single-Vehicle VRPP, which is formulated
    as a Prize-Collecting TSP (PC-TSP) with a global knapsack constraint.

    The model utilizes Generalized Subtour Elimination Constraints (GSECs).
    In the absence of explicit clusters, it primarily utilizes the Equation (2.3)
    form of GSECs: ∑_{(i,j) ∈ δ(S)} x[i,j] ≥ 2(yi + yj - 1).

    Uses the Natural Edge Formulation for Branch-and-Cut optimization.
    MTZ variables are omitted to preserve a tight polyhedral relaxation,
    relying strictly on the SeparationEngine for SEC and Capacity cuts.

    The VRPP seeks to maximize:
        Total Profit = (Waste Collected × R) - (Distance Traveled × C)

    Variable Consistency:
        The objective is implemented as a minimization of (Cost - Revenue)
        which is mathematically equivalent to maximizing (Revenue - Cost).

    Decision Variables:
        - x[i,j] ∈ {0,1,2}: Edge (i,j) usage. (ub=2 for edges incident to depot).
        - y[i] ∈ {0,1}: Node i is visited.

    Base Model Constraints (before cutting planes):
        1. Degree constraints: ∑_{(i,j) ∈ δ(i)} x[i,j] = 2·y[i] for all customers i
        2. Depot degree: ∑_{(0,j) ∈ δ(0)} x[0,j] ≤ 2K (where K is fleet size)
        3. Global capacity: ∑_{i ∈ V\\{0}} demand[i]·y[i] ≤ K·Q (aggregate capacity)
        4. Mandatory nodes: y[i] = 1 for i ∈ mandatory

    Dynamic Constraints (added via callbacks):
        5. Prize-Collecting Subtour Elimination Cuts (PC-SEC):
           - Form (2.1): ∑_{(i,j) ∈ δ(S)} x[i,j] ≥ 2
           - Form (2.2): ∑_{(i,j) ∈ δ(S)} x[i,j] ≥ 2yi
           - Form (2.3): ∑_{(i,j) ∈ δ(S)} x[i,j] ≥ 2(yi + yj - 1)
        6. Rounded Capacity Cuts (RCC): ∑_{(i,j) ∈ δ(S)} x[i,j] ≥ 2·⌈demand(S)/Q⌉
    """

    def __init__(
        self,
        n_nodes: int,
        cost_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        num_vehicles: int = 1,
        revenue_per_kg: float = 1.0,
        cost_per_km: float = 1.0,
        mandatory_nodes: Optional[Set[int]] = None,
        node_coords: Optional[Union[np.ndarray, Dict[int, Tuple[float, float]]]] = None,
    ):
        """
        Initialize the VRPP model.

        Args:
            n_nodes: Total number of nodes (including depot at index 0).
            cost_matrix: Symmetric n_nodes × n_nodes distance/cost matrix.
            wastes: Dictionary mapping node index -> waste/demand.
            capacity: Vehicle capacity constraint (Q).
            num_vehicles: Fleet size (K).
            revenue_per_kg: Revenue coefficient (R).
            cost_per_km: Cost coefficient (C).
            mandatory_nodes: Set of node indices that must be visited (mandatory).
            node_coords: Node coordinates for spatial branching.
        """
        self.n_nodes = n_nodes
        self.cost_matrix = cost_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.num_vehicles = num_vehicles
        self.R = revenue_per_kg
        self.C = cost_per_km
        self.mandatory_nodes = mandatory_nodes or set()
        self.node_coords = node_coords or {}

        # Depot is always node 0
        self.depot = 0
        self.customers = list(range(1, n_nodes))

        # Edge set (all symmetric pairs)
        self.edges: List[Tuple[int, int]] = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                self.edges.append((i, j))

        # Edge index mapping for quick lookup
        self.edge_to_idx: Dict[Tuple[int, int], int] = {}
        for idx, (i, j) in enumerate(self.edges):
            self.edge_to_idx[(i, j)] = idx
            self.edge_to_idx[(j, i)] = idx  # Symmetric

    def get_edge_cost(self, i: int, j: int) -> float:
        """Get the travel cost of edge (i, j)."""
        return self.cost_matrix[i, j] * self.C

    def get_node_profit(self, i: int) -> float:
        """Get the profit from visiting node i."""
        return self.wastes.get(i, 0.0) * self.R

    def get_node_demand(self, i: int) -> float:
        """Get the demand/waste at node i."""
        return self.wastes.get(i, 0.0)

    def delta(self, node_set: Set[int]) -> List[Tuple[int, int]]:
        """
        Return edges with exactly one endpoint in node_set (cut edges).

        δ(S) := {(i, j) ∈ E : i ∈ S, j ∉ S or i ∉ S, j ∈ S}
        """
        cut_edges = []
        for i, j in self.edges:
            if (i in node_set) != (j in node_set):  # XOR condition
                cut_edges.append((i, j))
        return cut_edges

    def edges_in_set(self, node_set: Set[int]) -> List[Tuple[int, int]]:
        """
        Return edges with both endpoints in node_set.

        E(S) := {(i, j) ∈ E : i ∈ S, j ∈ S}
        """
        edges_in = []
        for i, j in self.edges:
            if i in node_set and j in node_set:
                edges_in.append((i, j))
        return edges_in

    def total_demand(self, node_set: Set[int]) -> float:
        """Calculate total demand of nodes in node_set."""
        return sum(self.get_node_demand(i) for i in node_set if i != self.depot)

    def validate_tour(self, tour: List[int]) -> Tuple[bool, str]:
        """
        Validate that a tour satisfies VRPP constraints.

        Args:
            tour: Sequence of node indices representing the tour.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not tour or tour[0] != self.depot or tour[-1] != self.depot:
            return False, "Tour must start and end at depot (node 0)"

        # Check capacity constraint
        total_load = sum(self.get_node_demand(i) for i in tour[1:-1])
        if total_load > self.capacity:
            return False, f"Capacity violated: {total_load:.2f} > {self.capacity:.2f}"

        # Check mandatory nodes
        tour_set = set(tour)
        missing_mandatory = self.mandatory_nodes - tour_set
        if missing_mandatory:
            return False, f"Mandatory nodes not visited: {missing_mandatory}"

        # Check for cycles (no node visited more than once except depot)
        visited_counts: Dict[int, int] = {}
        for node in tour[1:-1]:  # Exclude depot returns
            visited_counts[node] = visited_counts.get(node, 0) + 1
            if visited_counts[node] > 1:
                return False, f"Node {node} visited {visited_counts[node]} times"

        return True, "Valid VRPP tour"

    def compute_tour_profit(self, tour: List[int]) -> float:
        """
        Compute the total profit of a tour.

        Profit = (Waste Collected × R) - (Distance Traveled × C)
        """
        # Waste collected (excluding depot, count each node once)
        tour_nodes = set(tour) - {self.depot}
        total_waste = sum(self.get_node_demand(i) for i in tour_nodes)
        revenue = total_waste * self.R

        # Distance traveled
        total_distance = sum(self.cost_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))
        travel_cost = total_distance * self.C

        return revenue - travel_cost

    def compute_tour_cost(self, tour: List[int]) -> float:
        """Compute the total travel distance/cost of a tour."""
        return sum(self.cost_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))
