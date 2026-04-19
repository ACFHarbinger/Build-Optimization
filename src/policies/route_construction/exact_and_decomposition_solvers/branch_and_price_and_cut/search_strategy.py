"""
Search strategies for Branch-and-Bound tree node selection in BPC algorithms.

Provides abstraction for different tree exploration strategies:
- Best-First Search (BFS): Explores nodes with best LP bounds first
- Depth-First Search (DFS): Explores deepest nodes first

References:
    - Barnhart, C., Hane, C. A., & Vance, P. H. (1998).
      "Using Branch-and-Price-and-Cut to Solve Origin-Destination Integer
      Multicommodity Flow Problems." Operations Research, 48(2), 318-326.
      (Uses DFS for ODIMCF to leverage LP basis warmstarts)

    - Achterberg, T., Koch, T., & Martin, A. (2005).
      "Branching rules revisited." Operations Research Letters, 33(1), 42-54.
      (Discusses various node selection strategies and their impact)
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from logic.src.policies.helpers.solvers_and_matheuristics import BranchAndBoundTree, BranchNode
else:
    # Use Any for runtime or environments where TYPE_CHECKING is False
    BranchAndBoundTree = Any
    BranchNode = Any

logger = logging.getLogger(__name__)


class NodeSelectionStrategy(ABC):
    """
    Abstract base class for B&B tree node selection strategies.

    Different strategies prioritize nodes differently, affecting both the
    search performance and the ability to leverage LP basis information.
    """

    @abstractmethod
    def select_node(self, open_nodes: List[BranchNode]) -> Optional[BranchNode]:
        """
        Select and remove the next node to explore from the open node list.

        Args:
            open_nodes: List of unexplored branch nodes

        Returns:
            Selected node, or None if list is empty
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the strategy name for logging and debugging."""
        pass


class BestFirstSearch(NodeSelectionStrategy):
    """
    Best-First Search strategy: selects node with highest LP objective bound.

    While standard for various optimization problems, in the context of BPC
    for VRPP, it can be less efficient than Depth-First Search because it
    frequently jumps across different tree branches, preventing effective
    LP basis reuse.

    Characteristics:
    - Time complexity: O(n) per selection (linear scan for maximum)
    - Space complexity: O(n) where n is number of open nodes
    - Optimality guarantee: Yes (like all complete B&B strategies)
    - Basis reuse: Poor (jumps across different tree branches)

    Best used for:
    - Small VRPP instances where tree depth is limited.
    - Situations where proving optimality with the fewest nodes is
      prioritized over computational speed per node.
    """

    def select_node(self, open_nodes: List[BranchNode]) -> Optional[BranchNode]:
        """
        Select node with highest LP bound (maximization problem).

        For maximization problems (VRPP profit maximization), we want
        to explore nodes with the highest upper bound first.

        Args:
            open_nodes: List of unexplored branch nodes

        Returns:
            Node with highest LP bound, or None if list is empty
        """
        if not open_nodes:
            return None

        # Fix 8: Use O(n) scan instead of O(n log n) sort
        def get_best_first_key(i: int) -> float:
            bound = open_nodes[i].lp_bound
            return bound if bound is not None else float("-inf")

        best_idx = max(range(len(open_nodes)), key=get_best_first_key)

        return open_nodes.pop(best_idx)

    def get_name(self) -> str:
        return "best_first"


class DepthFirstSearch(NodeSelectionStrategy):
    """
    Depth-First Search strategy: selects deepest node in the tree.

    This is the preferred default strategy for the BPC solver as it:
    - Maximizes LP basis reuse by staying within the same branch.
    - Achieves significant speedups (often 3-5x) by resolving child LPs
      from the parent's basis using dual simplex.
    - Effectively leverages LP warm-starts as advocated by Barnhart et al. (1998, OR 46(3):316-329).

    Characteristics:
    - Time complexity: O(n) per selection (linear scan)
    - Space complexity: O(n) where n is number of open nodes
    - Optimality guarantee: Yes (explores entire tree if needed)
    - Basis reuse: Excellent (stays within same branch until fathomed)

    Best used for:
    - High-performance BPC on VRPP where LP solve time dominates.
    - Large instances where memory management of Basis information
      for Best-First search would be prohibitive.

    Implementation Note:
    ----------------------
    When combined with LP warmstarts, DFS can achieve 3-5x speedup on
    large multicommodity flow instances by reusing the basis from the
    parent node (Barnhart et al., 1998, OR 46(3):316-329, Section 5).

    The solver should leverage Gurobi's or CPLEX's basis warmstart
    capabilities by:
    1. Storing the basis from the parent node's final LP solve
    2. Loading this basis before solving the child node's LP
    3. Using dual simplex to resolve from the warmstart
    """

    def select_node(self, open_nodes: List[BranchNode]) -> Optional[BranchNode]:
        """
        Select the next node to explore using DFS (LIFO).

        Args:
            open_nodes: List of unexplored branch nodes (treated as a stack).

        Returns:
            The last node appended, or None if the list is empty.
        """
        if not open_nodes:
            return None
        # open_nodes is used as a LIFO stack: the last element appended is
        # the next to explore. In run_custom_bpc, add the preferred branch
        # (shorter path / higher LP bound) LAST so it is popped first.
        return open_nodes.pop()

    def get_name(self) -> str:
        return "depth_first"


class HybridSearchStrategy(NodeSelectionStrategy):
    """
    Hybrid Dive-and-Best-Bound (D&BB) strategy.

    Logic:
    1. If no integer incumbent exists (best_integer_solution is None):
       Behaves like DFS (LIFO). Rapidly dives to find a feasible solution.
    2. Once an incumbent exists:
       Behaves like BestFS (Best-Bound). Systematic closure of the gap.

    This strategy combines the speed of DFS in finding feasible solutions with
    the mathematical efficiency of BestFS in proving optimality.

    NOTE ON PERFORMANCE:
    Hybrid search is less efficient than pure DFS for resolving child LPs
    because the Best-Bound phase frequently jumps between branches, forcing
    frequent Basis invalidation and re-solving from scratch. It also occupies
    more memory than DFS as Basis information must be stored for all open
    nodes to maintain any hope of warm-starting.
    """

    def __init__(self, bb_tree: "BranchAndBoundTree"):
        """
        Store reference to tree state to monitor incumbents.

        Args:
            bb_tree: The B&B tree structure.
        """
        self.bb_tree = bb_tree

    def select_node(self, open_nodes: List[BranchNode]) -> Optional[BranchNode]:
        """
        Select node based on search phase.

        Args:
            open_nodes: List of unexplored branch nodes.

        Returns:
            Selected node.
        """
        if not open_nodes:
            return None

        # Dive Phase: DFS (LIFO)
        if self.bb_tree.best_integer_solution is None:
            return open_nodes.pop()

        # Bound Phase: Best-First Search (O(n) scan)
        def get_best_first_key(i: int) -> float:
            bound = open_nodes[i].lp_bound
            return bound if bound is not None else float("-inf")

        best_idx = max(range(len(open_nodes)), key=get_best_first_key)
        return open_nodes.pop(best_idx)

    def get_name(self) -> str:
        return "hybrid"


def create_search_strategy(strategy_name: str, bb_tree: Optional["Any"] = None) -> NodeSelectionStrategy:
    """
    Factory function to create node selection strategies.

    Args:
        strategy_name: Name of the strategy ("best_first", "depth_first", "hybrid")
        bb_tree: Optional reference to the tree for hybrid strategy.

    Returns:
        Instance of the requested strategy

    Raises:
        ValueError: If strategy_name is not recognized
    """
    if strategy_name == "best_first":
        return BestFirstSearch()
    elif strategy_name == "depth_first":
        return DepthFirstSearch()
    elif strategy_name == "hybrid":
        if bb_tree is None:
            raise ValueError("Hybrid search strategy requires a BranchAndBoundTree reference.")
        return HybridSearchStrategy(bb_tree)
    else:
        raise ValueError(
            f"Unknown search strategy '{strategy_name}'. Valid options are: ['best_first', 'depth_first', 'hybrid']"
        )
