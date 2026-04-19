"""
Branch-and-Bound tree management for VRPP.
"""

from __future__ import annotations

import heapq
import logging
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics.branching.strategies import (
    EdgeBranching,
    FleetSizeBranching,
    MultiEdgePartitionBranching,
    NodeVisitationBranching,
    RyanFosterBranching,
)
from logic.src.policies.helpers.solvers_and_matheuristics.common.node import BranchNode

if TYPE_CHECKING:
    from logic.src.policies.helpers.solvers_and_matheuristics.common.route import Route
    from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel
    from logic.src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.params import (
        BPCParams,
    )

# The type alias will now use the globally unified BranchNode
FrontierItem = Union[BranchNode, Tuple[float, int, BranchNode]]


class BranchAndBoundTree:
    """
    Manages the search tree and node frontier for Branch-and-Price-and-Cut.

    Synthesizes global search control logic (frontier queueing, incumbent tracking)
    with advanced, pluggable branching strategies (Spatial Divergence, Ryan-Foster, Edge).

    Supports multiple strategies via the ``strategy`` constructor parameter:

    ``"edge"``
        Branches on the most-fractional directed arc using
        :class:`EdgeBranching`.  Integrates natively with the DP pricing
        subproblem.

    ``"ryan_foster"``
        Branches on a fractional node-pair co-occurrence using
        :class:`RyanFosterBranching`.  Compatible with set-partitioning
        master problems.

    Theoretical Context:
    The tree utilizes the 'best-bound' priority for Best-First search to
    minimize the number of nodes explored before proving optimality. This
    is particularly effective for VRPP where the root LP gap is often
    small (~1-3%), allowing for early pruning.
    """

    def __init__(
        self,
        v_model: Optional[VRPPModel] = None,
        params: Optional[BPCParams] = None,
        # Legacy positional arguments kept for backward compatibility only.
        # If params is supplied these are ignored; a DeprecationWarning is emitted
        # when they differ from their defaults to surface accidental misuse.
        max_nodes: int = 1000,
        strategy: str = "edge",
        search_strategy: str = "best_first",
    ) -> None:
        """
        Initialize the Branch-and-Bound tree for BPC.

        Args:
            v_model: The underlying VRPP or Master problem model.
            params: Standardized B&B parameters.
            max_nodes: Maximum number of nodes to explore.
            strategy: Branching strategy ('divergence_spatial', 'edge', 'ryan_foster').
            search_strategy: Search strategy ('best_first', 'depth_first').
        """
        if params is not None:
            if max_nodes != 1000 or strategy != "edge" or search_strategy != "best_first":
                warnings.warn(
                    "BranchAndBoundTree: explicit 'max_nodes', 'strategy', and "
                    "'search_strategy' arguments are ignored when 'params' is supplied. "
                    "Configure these via BPCParams instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            self.max_nodes = getattr(params, "max_branch_nodes", max_nodes)
            self.strategy = getattr(params, "branching_strategy", strategy)
            self.search_strategy = getattr(params, "tree_search_strategy", search_strategy)
        else:
            self.max_nodes = max_nodes
            self.strategy = strategy
            self.search_strategy = search_strategy

        self.v_model = v_model
        self.node_coords: Optional[np.ndarray] = None

        # Extract coordinates from the injected model
        if v_model and hasattr(v_model, "node_coords"):
            node_coords = v_model.node_coords
            if isinstance(node_coords, dict):
                coords_arr = np.zeros((len(node_coords) + 1, 2))
                for i, (x, y) in node_coords.items():
                    coords_arr[i] = [x, y]
                self.node_coords = coords_arr
            else:
                self.node_coords = node_coords

        self.nodes_explored = 0
        self.nodes_pruned = 0
        self.best_integer_solution: Optional[float] = None
        self.best_integer_node: Optional[BranchNode] = None

        self.open_nodes: List[FrontierItem] = []
        self._counter = 0

        self.root = BranchNode()
        self.add_node(self.root)

    # ------------------------------------------------------------------
    # Frontier Management
    # ------------------------------------------------------------------

    def add_node(self, node: BranchNode) -> None:
        """Enqueue a new open node to the frontier."""
        if self.search_strategy in ("best_first", "best-first"):
            # Maximize LP bound, but heapq is a min-heap, so we negate the bound.
            priority = -(node.lp_bound if node.lp_bound is not None else 1e9)
            heapq.heappush(self.open_nodes, (priority, self._counter, node))  # type: ignore[misc]
            self._counter += 1
        else:
            # Simple list for Depth-First Search (LIFO)
            self.open_nodes.append(node)

    def get_next_node(self) -> Optional[BranchNode]:
        """
        Select and remove the next node to process from the open list.

        Uses the configured search strategy:
        - "best_first": pop the node with the highest LP bound (best-bound-first).
        - "depth_first": pop the most recently added node (LIFO).

        Returns:
            The selected BranchNode, or None if the open list is empty.
        """
        if not self.open_nodes:
            return None
        if self.search_strategy in ("best_first", "best-first"):
            _, _, node = heapq.heappop(self.open_nodes)  # type: ignore[misc]
            return node  # type: ignore[has-type]
        else:
            return self.open_nodes.pop()  # type: ignore[return-value]

    def is_empty(self) -> bool:
        """Return True when no open nodes remain in the frontier."""
        return len(self.open_nodes) == 0

    # ------------------------------------------------------------------
    # Pruning and Incumbent Management
    # ------------------------------------------------------------------

    def update_incumbent(self, node: BranchNode, value: float) -> bool:
        """
        Update the best known integer solution if *value* improves it.

        Args:
            node: Node where the integer solution was found.
            value: Objective value of the integer solution.

        Returns:
            True if the incumbent was improved.
        """
        if self.best_integer_solution is None or value > self.best_integer_solution + 1e-6:
            self.best_integer_solution = value
            self.best_integer_node = node
            logging.info(f"New global LB (Incumbent): {self.best_integer_solution:.4f}")
            return True
        return False

    def prune_by_bound(self) -> int:
        """
        Remove nodes from the frontier whose LP bound cannot improve the incumbent.

        Reference: Barnhart et al. (1998) pruning rules based on the
        current integer incumbent and the node's LP relaxation bound.

        Returns:
            Number of nodes pruned.
        """
        if self.best_integer_solution is None:
            return 0

        before = len(self.open_nodes)
        limit = self.best_integer_solution + 1e-8

        if self.search_strategy in ("best_first", "best-first"):
            # Item is (priority, counter, node), where priority = -lp_bound
            # We cast self.open_nodes to the specific tuple list type for this block
            best_first_nodes = cast(List[Tuple[float, int, BranchNode]], self.open_nodes)

            self.open_nodes = [
                (prio, count, n) for prio, count, n in best_first_nodes if not n.is_infeasible and (-prio > limit)
            ]
            heapq.heapify(self.open_nodes)
        else:
            # DFS/LIFO mode uses a simple list of BranchNodes
            dfs_nodes = cast(List[BranchNode], self.open_nodes)

            self.open_nodes = [
                n for n in dfs_nodes if not n.is_infeasible and (n.lp_bound is None or n.lp_bound > limit)
            ]

        pruned = before - len(self.open_nodes)
        self.nodes_pruned += pruned
        if pruned > 0:
            logging.info(f"Pruned {pruned} nodes from frontier by bound.")
        return pruned

    def record_explored(self) -> None:
        """Increment the global nodes explored counter."""
        self.nodes_explored += 1

    # ------------------------------------------------------------------
    # Branching
    # ------------------------------------------------------------------

    def find_strong_branching_candidates(
        self, routes: List["Route"], route_values: Dict[int, float], max_candidates: int = 5
    ) -> List[Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]], float]]:
        """
        Task 11 (SOTA): Identify top branching candidates for lookahead eval.
        Uses Spatial Divergence strength as the primary heuristic.
        """
        candidates = []
        n_nodes = self.v_model.n_nodes - 1 if self.v_model and hasattr(self.v_model, "n_nodes") else 0

        div_candidates = MultiEdgePartitionBranching.find_multiple_divergence_nodes(
            routes,
            route_values,
            node_coords=self.node_coords,
            limit=max_candidates,
            n_nodes=n_nodes,
        )
        for cand in div_candidates:
            d, arcs1, arcs2, score = cand
            candidates.append((d, arcs1, arcs2, score))

        # Sort by fractional score (how close flow is to 0.5)
        candidates.sort(key=lambda x: abs(0.5 - (x[3] % 1.0)), reverse=True)
        return candidates[:max_candidates]

    def branch(
        self,
        node: BranchNode,
        routes: List["Route"],
        route_values: Dict[int, float],
        mandatory_nodes: Set[int],
        strong_candidate: Optional[Any] = None,
    ) -> Optional[Tuple[BranchNode, BranchNode]]:
        """
        Apply the active branching strategy and return two child nodes.

        This is the single dispatch point for branching logic.  Callers no
        longer need to import strategy classes directly.

        Args:
            node: The fractional B&B node to branch from.
            routes: All routes in the master problem at this node.
            route_values: Current LP solution {route_index: λ_k}.

        Returns:
            ``(left_child, right_child)`` if a branching decision was found,
            or ``None`` if the solution is already integer (no fractional
            variable / arc found).
        """
        # 1. Strong Branching candidate override
        if strong_candidate:
            d, arcs1, arcs2, _ = strong_candidate
            return MultiEdgePartitionBranching.create_child_nodes(node, d, arcs1, arcs2)

        # 2. Level 1: Fleet Size branching
        res_fleet = FleetSizeBranching.find_fleet_branching(route_values)
        if res_fleet:
            return FleetSizeBranching.create_child_nodes(node, res_fleet)

        # 3. Level 2: Divergence branching (Preferred spatial rule)
        n_nodes = self.v_model.n_nodes - 1 if self.v_model and hasattr(self.v_model, "n_nodes") else 0
        res_div = MultiEdgePartitionBranching.find_divergence_node(
            routes,
            route_values,
            node_coords=self.node_coords,
            n_nodes=n_nodes,
        )
        if res_div:
            d, arcs1, arcs2, _ = res_div
            return MultiEdgePartitionBranching.create_child_nodes(node, d, arcs1, arcs2)

        # 4. Level 3: Ryan-Foster branching (co-occurrence)
        res_rf = RyanFosterBranching.find_branching_pair(routes, route_values, mandatory_nodes)
        if res_rf:
            pair, together_sum = res_rf
            return RyanFosterBranching.create_child_nodes(node, pair[0], pair[1], together_sum)

        # 5. Level 4: Simple Edge Branching
        res_edge = EdgeBranching.find_branching_arc(routes, route_values)
        if res_edge is not None:
            arc, flow = res_edge
            return EdgeBranching.create_child_nodes(node, arc[0], arc[1], flow)

        # 6. Level 5: Node Visitation (Last Resort - Mandatory only)
        if mandatory_nodes:
            node_frac_res = NodeVisitationBranching.find_node_branching(routes, route_values, mandatory_nodes)
            if node_frac_res is not None:
                n, visitation = node_frac_res
                return NodeVisitationBranching.create_child_nodes(node, n, visitation)

        return None

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return a snapshot of tree-search statistics.

        This method aggregates global search metadata, including the current
        best upper bound (from open nodes) and the best integer lower bound
        found so far.
        """
        best_bound: Optional[float] = None
        if self.open_nodes:
            if self.search_strategy in ("best_first", "best-first"):
                best_first_nodes = cast(List[Tuple[float, int, BranchNode]], self.open_nodes)
                # Priority is -lp_bound; negate to recover actual bound
                best_bound = max((-prio for prio, _, _ in best_first_nodes), default=None)
            else:
                dfs_nodes = cast(List[BranchNode], self.open_nodes)
                best_bound = max((n.lp_bound for n in dfs_nodes if n.lp_bound is not None), default=None)

        return {
            "nodes_explored": self.nodes_explored,
            "nodes_pruned": self.nodes_pruned,
            "nodes_remaining": len(self.open_nodes),
            "best_bound": best_bound,
            "best_integer": self.best_integer_solution,
            "branching_strategy": self.strategy,
        }
