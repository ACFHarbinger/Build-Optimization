from typing import List, Optional, Set

"""
HGS Individual Module.

This module defines the Individual class used in the Hybrid Genetic Search (HGS)
algorithm. It represents a single solution in the population, storing genotype
(giant tour) and phenotype (routes), along with cost and fitness metrics.

Attributes:
    None

Example:
    >>> from logic.src.policies.hybrid_genetic_search.individual import Individual
    >>> ind = Individual(genotype=[1, 5, 2, ...], cost=100.0)
"""


class Individual:
    """
    Individual solution representation for genetic algorithm.
    Follows Vidal et al. (2022) HGS-CVRP implementation with VRPP adaptations.

    Genotype/Phenotype Mapping:
        - Genotype: giant_tour (ordered list of all customer nodes, including unvisited)
        - Phenotype: routes (actual vehicle routes after Split algorithm)

    For VRPP (Vehicle Routing Problem with Profits):
        The Split algorithm may skip unprofitable nodes, creating a mismatch between
        genotype and phenotype. To maintain genetic consistency for crossover:

        1. giant_tour contains ALL nodes in a specific order (genotype)
        2. routes contains only visited nodes organized into vehicle routes (phenotype)
        3. Nodes in giant_tour but not in routes are implicitly "unvisited"
        4. Local search operators may move nodes between visited and unvisited sets
        5. Crossover operates on giant_tour, preserving genetic material for all nodes

    This approach (Option A: Implicit Dummy Route) treats unvisited nodes as being
    in a conceptual "dummy route" at the end of the giant tour.
    """

    def __init__(self, giant_tour: List[int], expand_pool: bool = False):
        """
        Initialize an individual with a giant tour.

        Args:
            giant_tour: A list representing the visit order of all clients (genotype).
                       For VRPP, this includes both visited and potentially unvisited nodes.
            expand_pool: If True (VRPP mode), repair operators consider all unvisited nodes.
                        If False (CVRP mode), repair operators only consider removed nodes.
        """
        self.giant_tour = giant_tour
        self._routes: List[List[int]] = []
        self._visited_cache: Optional[Set[int]] = None
        # Fix 8: Fitness is minimized, so initialize to +inf.
        self.fitness = float("inf")
        self.profit_score = -float("inf")
        self.cost = 0.0
        self.revenue = 0.0

        # Feasibility tracking
        self.is_feasible = True
        self.capacity_violation = 0.0  # Total capacity excess across all routes
        self.penalized_cost = 0.0  # Cost including penalty for violations

        # Diversity and ranking
        self.dist_to_parents = 0.0
        self.rank_profit = 0
        self.rank_diversity = 0
        self.is_coached = False

        # VRPP-specific: controls whether repair operators expand node pool
        self.expand_pool = expand_pool

    @property
    def routes(self) -> List[List[int]]:
        """Actual vehicle routes after Split algorithm (phenotype)."""
        return self._routes

    @routes.setter
    def routes(self, value: List[List[int]]) -> None:
        """Set routes and invalidate visited nodes cache."""
        self._routes = value
        self._visited_cache = None

    @property
    def penalized_profit(self) -> float:
        """
        Revenue minus penalized cost. Used for quality ranking in HGS.
        (Fix 10: Named property for penalised profit expression)
        """
        # penalized_cost = cost + penalty * violation
        # profit_score = revenue - cost
        # revenue - cost - penalty * violation = profit_score - penalized_cost + cost
        return self.profit_score - self.penalized_cost + self.cost

    def get_visited_nodes(self) -> Set[int]:
        """
        Get the set of nodes that are currently visited in the solution.

        Returns:
            Set of node IDs present in routes (phenotype).
        """
        # Fix 9: Cache visited nodes for performance in tight local search loops.
        if self._visited_cache is None:
            self._visited_cache = {node for route in self._routes for node in route}
        return self._visited_cache

    def get_unvisited_nodes(self) -> List[int]:
        """
        Get the list of nodes in the giant tour that are not currently visited.

        For VRPP, these are nodes that the Split algorithm decided to skip
        because they were unprofitable, or that local search removed.

        Returns:
            List of node IDs in giant_tour but not in routes, in giant_tour order.
        """
        visited = self.get_visited_nodes()
        return [node for node in self.giant_tour if node not in visited]

    def assert_invariants(self, expected_nodes: Optional[Set[int]] = None) -> None:
        """
        Assert that the individual satisfies HGS genotype invariants:

        1. giant_tour contains no duplicate nodes.
        2. All nodes in routes are also in giant_tour.
        3. If expected_nodes is provided, giant_tour contains exactly those nodes.
        (Fix 21: Add a genotype integrity check utility)
        """
        tour_set = set(self.giant_tour)
        assert len(tour_set) == len(self.giant_tour), (
            f"giant_tour contains duplicates: {len(self.giant_tour) - len(tour_set)} duplicate(s)."
        )

        for route in self._routes:
            for node in route:
                assert node in tour_set, f"Node {node} appears in routes but not in giant_tour."

        if expected_nodes is not None:
            assert tour_set == expected_nodes, (
                f"giant_tour node set mismatch. "
                f"Missing: {expected_nodes - tour_set}. "
                f"Extra: {tour_set - expected_nodes}."
            )

    def __lt__(self, other: "Individual") -> bool:
        """Comparison based on biased fitness (used for sorting)."""
        return self.fitness < other.fitness
