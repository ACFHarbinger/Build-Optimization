"""
Branching strategies and heuristics for VRPP Column Generation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics.branching.constraints import (
    EdgeBranchingConstraint,
    FleetSizeBranchingConstraint,
    NodeVisitationBranchingConstraint,
    RyanFosterBranchingConstraint,
)
from logic.src.policies.helpers.solvers_and_matheuristics.common.route import Route

if TYPE_CHECKING:
    from logic.src.policies.helpers.solvers_and_matheuristics.common.node import BranchNode


class EdgeBranching:
    """
    Edge-based branching: select the most-fractional arc and split on it.

    Produces two child nodes:
        left  → x_{uv} = 1  (arc MUST be used)
        right → x_{uv} = 0  (arc is FORBIDDEN)
    """

    @staticmethod
    def compute_arc_flow(
        routes: List[Route],
        route_values: Dict[int, float],
    ) -> Dict[Tuple[int, int], float]:
        """
        Aggregate fractional arc flows from the LP solution.

        Args:
            routes: All routes in the master problem column pool.
            route_values: LP solution values {route_index: λ_k}.

        Returns:
            Mapping (u, v) → aggregated fractional flow.
        """
        arc_flow: Dict[Tuple[int, int], float] = {}
        for idx, lam in route_values.items():
            if lam < 1e-9:
                continue
            full_path = [0] + routes[idx].nodes + [0]
            for i in range(len(full_path) - 1):
                arc = (full_path[i], full_path[i + 1])
                arc_flow[arc] = arc_flow.get(arc, 0.0) + lam
        return arc_flow

    @staticmethod
    def find_branching_arc(
        routes: List[Route],
        route_values: Dict[int, float],
        tol: float = 1e-5,
    ) -> Optional[Tuple[Tuple[int, int], float]]:
        """
        Select the arc with fractional flow closest to 0.5.

        Args:
            routes: All routes in the master problem.
            route_values: Current LP solution {route_index: λ_k}.
            tol: Integrality tolerance.

        Returns:
            Arc (u, v) to branch on, or None if the solution is already integer.
        """
        arc_flow = EdgeBranching.compute_arc_flow(routes, route_values)
        best_arc: Optional[Tuple[int, int]] = None
        best_frac = -1.0
        best_arc_flow = 0.0

        for arc, flow in arc_flow.items():
            frac = min(flow, 1.0 - flow)
            if frac > tol and frac > best_frac:
                best_frac = frac
                best_arc = arc
                best_arc_flow = flow

        if best_arc is None:
            return None
        return best_arc, best_arc_flow

    @staticmethod
    def create_child_nodes(
        parent: "BranchNode",
        u: int,
        v: int,
        flow: float = 0.5,
    ) -> Tuple["BranchNode", "BranchNode"]:
        """
        Create left (must-use) and right (forbidden) child nodes.

        Args:
            parent: The node being branched.
            u: Arc origin.
            v: Arc destination.

        Returns:
            (left_child, right_child)
        """
        from logic.src.policies.helpers.solvers_and_matheuristics.common.node import BranchNode

        left_bc = EdgeBranchingConstraint(u, v, must_use=True)
        right_bc = EdgeBranchingConstraint(u, v, must_use=False)

        return (
            BranchNode(
                constraints=[left_bc],
                parent=parent,
                depth=parent.depth + 1,
                lp_bound_hint=flow,
                branching_rule="edge",
            ),
            BranchNode(
                constraints=[right_bc],
                parent=parent,
                depth=parent.depth + 1,
                lp_bound_hint=1.0 - flow,
                branching_rule="edge",
            ),
        )


class MultiEdgePartitionBranching:
    r"""
    Advanced Divergence Branching with Spatial Fleet Partitioning.

    This strategy formalizes the Divergence Branching of Barnhart et al. (1998)
    by utilizing node coordinates to induce a spatially cohesive arc-set
    partition. Unlike ODIMCF which restricts specific vehicles (commodities),
    this restricts the entire anonymous fleet, making it polyhedrally stronger.

    Mathematical Formulation:
    -------------------------
    1. Divergence Node Identification:
       Identify a 'divergence node' $d$ where the fractional flow $\bar{x}$
       diverges into multiple outgoing arcs.
       $A_d^+ = \{(d, v) \in E : 0 < \bar{x}_{dv} < 1\}$

    2. Polar Mapping:
       Define a spatial mapping function $f(v) = \operatorname{atan2}(y_v - y_d, x_v - x_d)$
       returning the polar angle of node $v$ relative to $d$.

    3. Spatial Partitioning:
       Sort $A_d^+$ by destination polar angles and partition into $S_1$ and $S_2$
       via a median split. This creates two balanced geographic sectors.
       $\mathcal{L}: \sum_{(d,v) \in S_1} x_{dv} = 0 \quad \text{and} \quad \mathcal{R}: \sum_{(d,v) \in S_2} x_{dv} = 0$

    4. Candidate Scoring (SVRPC):
       Candidates are ranked by the Spatial Variable Routing Persistence (SVRP)
       strength, calculated as the fractional flow persistence across the split:
       $\sigma(d) = \sum_{(d,v) \in S_1} \bar{x}_{dv}$

    Theoretical Rationale:
    ----------------------
    Spatial fleet partitioning separates the routing topology into convex
    geographic polygons. By forbidding arc sets rather than single edges,
    it globally restricts the anonymous fleet, enforcing a strong bound.
    """

    @staticmethod
    def find_divergence_node(  # noqa: C901
        routes: List[Route],
        route_values: Dict[int, float],
        tol: float = 1e-5,
        node_coords: Optional[Union[np.ndarray, Dict[int, Tuple[float, float]]]] = None,
        n_nodes: int = 0,
    ) -> Optional[Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]], float]]:
        r"""
        Barnhart, Hane, and Vance (2000) §4.3 path-tracing divergence branching.

        Primary method — explicit path tracing (paper §4.3):
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Selects the two highest-λ fractional routes (P1, P2) and walks them
        in lock-step from the depot until they take different outgoing arcs
        from some node d.  That node is the *divergence node*.

        Arc-set convention (DFS child ordering):
          arc_set_1  contains the higher-flow half of a balanced median split of
                     all outgoing fractional arcs from d (by aggregate λ weight).
          arc_set_2  contains the lower-flow half plus all non-fractional outgoing
                     arcs from d, guaranteeing arc-universe completeness.

        Child 1 (forbids arc_set_1) → Typically the "weaker" branch, explored
                 FIRST in DFS to find an early feasible solution.
        Child 2 (forbids arc_set_2) → Typically the "stronger" branch, explored
                 SECOND, focusing search on the higher-flow arc cluster.

        This balanced partition prevents "thin" branches and improves tree
        convergence compared to singleton arc branching.

        Fallback — aggregate-flow heuristic with optional spatial partition:
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Used when fewer than two fractional routes exist, or when the two
        chosen routes share no common initial sub-path long enough to reveal
        a non-trivial divergence node.  Falls back to the arc with the
        highest aggregate flow, then uses cosine-similarity spatial grouping
        to build a balanced arc-set partition.

        Arc-universe completeness guarantee:
        arc_set_1 ∪ arc_set_2 = all possible outgoing arcs from d, so no
        new column can bypass the branching constraint in either child.

        Args:
            routes: Column pool.
            route_values: Fractional LP solution {route_index: λ_k}.
            tol: Integrality tolerance.
            node_coords: Optional (N+1)×2 coordinate array; used only in
                the fallback spatial partition.
            n_nodes: Number of customer nodes (excluding depot); used to
                build the complete outgoing-arc universe.

        Returns:
            (d, arc_set_1, arc_set_2, score) or None if solution is integer.
        """
        fractional: List[Tuple[float, int]] = [(lam, idx) for idx, lam in route_values.items() if tol < lam < 1.0 - tol]

        if len(fractional) >= 2:
            fractional.sort(reverse=True)
            lam1, idx1 = fractional[0]
            lam2, idx2 = fractional[1]

            path1: List[int] = [0] + routes[idx1].nodes + [0]
            path2: List[int] = [0] + routes[idx2].nodes + [0]

            divergence_d: Optional[int] = None
            arc_p1: Optional[Tuple[int, int]] = None
            arc_p2: Optional[Tuple[int, int]] = None

            for pos in range(min(len(path1), len(path2)) - 1):
                if path1[pos] != path2[pos]:
                    break
                d_cand = path1[pos]
                nxt1, nxt2 = path1[pos + 1], path2[pos + 1]
                if nxt1 != nxt2:
                    divergence_d = d_cand
                    arc_p1 = (d_cand, nxt1)
                    arc_p2 = (d_cand, nxt2)
                    break

            if divergence_d is not None and arc_p1 is not None and arc_p2 is not None:
                if len(path1) > len(path2):
                    arc_p1, arc_p2 = arc_p2, arc_p1

                if n_nodes > 0:
                    node_univ: Set[int] = set(range(n_nodes + 1))
                else:
                    node_univ = {0}
                    for r in routes:
                        node_univ.update(r.nodes)
                node_univ.discard(divergence_d)
                all_out: Set[Tuple[int, int]] = {(divergence_d, v) for v in node_univ}

                d_out_flows: Dict[int, float] = {}
                for idx, lam in route_values.items():
                    if lam < tol:
                        continue
                    path = [0] + routes[idx].nodes + [0]
                    for p in range(len(path) - 1):
                        if path[p] == divergence_d:
                            v_nxt = path[p + 1]
                            d_out_flows[v_nxt] = d_out_flows.get(v_nxt, 0.0) + lam

                nxt_p1 = arc_p1[1]
                sorted_v = sorted(
                    d_out_flows.keys(),
                    key=lambda v: (v == nxt_p1, d_out_flows[v]),
                    reverse=True,
                )
                total_f = sum(d_out_flows.values())
                acc = 0.0
                split_idx = 0
                for i, v in enumerate(sorted_v):
                    acc += d_out_flows[v]
                    if acc >= total_f / 2.0:
                        split_idx = i + 1
                        break

                s1_nodes = set(sorted_v[: max(1, split_idx)])
                arc_set_1_pt = [(divergence_d, v) for v in s1_nodes]
                arc_set_2_pt = sorted(all_out - set(arc_set_1_pt))

                score_pt = sum(d_out_flows[v] for v in s1_nodes)
                return divergence_d, arc_set_1_pt, arc_set_2_pt, score_pt

        # 3. Fallback: aggregate-flow heuristic with optional spatial
        node_out_flows: Dict[int, Dict[int, float]] = {}
        for idx, lam in route_values.items():
            if lam < tol:
                continue
            r = routes[idx]
            full_path = [0] + r.nodes + [0]
            for i in range(len(full_path) - 1):
                u, v = full_path[i], full_path[i + 1]
                node_out_flows.setdefault(u, {}).setdefault(v, 0.0)
                node_out_flows[u][v] += lam

        div_candidates = [d for d, flows in node_out_flows.items() if len(flows) >= 2]
        if not div_candidates:
            return None

        d = max(div_candidates, key=lambda x: sum(node_out_flows[x].values()))

        if n_nodes > 0:
            node_universe_fb: Set[int] = set(range(n_nodes + 1))
        else:
            node_universe_fb = {0}
            for r in routes:
                node_universe_fb.update(r.nodes)
        node_universe_fb.discard(d)
        all_possible_outgoing: Set[Tuple[int, int]] = {(d, v) for v in node_universe_fb}

        flows_fb = node_out_flows[d]
        v_sorted_fb = sorted(flows_fb.keys(), key=lambda v: flows_fb[v], reverse=True)
        total_f_fb = sum(flows_fb.values())
        acc_fb = 0.0
        split_idx_fb = 0
        for i, v in enumerate(v_sorted_fb):
            acc_fb += flows_fb[v]
            if acc_fb >= total_f_fb / 2.0:
                split_idx_fb = i + 1
                break

        s1_nodes_fb = set(v_sorted_fb[: max(1, split_idx_fb)])
        arc_set_1_fb: Set[Tuple[int, int]] = {(d, v) for v in s1_nodes_fb}
        v1 = v_sorted_fb[0]

        if node_coords is not None:
            d_coord = np.array(node_coords[d])
            v1_coord = np.array(node_coords[v1])
            vec_v1 = v1_coord - d_coord
            norm_v1 = float(np.linalg.norm(vec_v1))
            if norm_v1 > 1e-9:
                for v in node_universe_fb:
                    if v == v1:
                        continue
                    v_coord = np.array(node_coords[v])
                    vec_v = v_coord - d_coord
                    norm_v = float(np.linalg.norm(vec_v))
                    if norm_v < 1e-9:
                        continue
                    similarity = float(np.dot(vec_v1, vec_v)) / (norm_v1 * norm_v)
                    if similarity > 0.5:
                        arc_set_1_fb.add((d, v))

        arc_set_2_fb = all_possible_outgoing - arc_set_1_fb
        score_fb = sum(flows_fb[v] for v in flows_fb if (d, v) in arc_set_1_fb)
        return d, list(arc_set_1_fb), list(arc_set_2_fb), score_fb

    @staticmethod
    def find_multiple_divergence_nodes(
        routes: List[Route],
        route_values: Dict[int, float],
        node_coords: Optional[Union[np.ndarray, Dict[int, Tuple[float, float]]]] = None,
        limit: int = 5,
        tol: float = 1e-5,
        n_nodes: int = 0,
    ) -> List[Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]], float]]:
        """
        Implement proper multi-candidate support (Fix 19).
        """
        candidates = []
        remaining_route_values = dict(route_values)

        for _ in range(limit):
            if len(remaining_route_values) < 2:
                break
            res = MultiEdgePartitionBranching.find_divergence_node(
                routes, remaining_route_values, tol, node_coords, n_nodes
            )
            if res is None:
                break
            candidates.append(res)
            d, arc_set_1, arc_set_2, _ = res
            remaining_route_values = {
                idx: lam for idx, lam in remaining_route_values.items() if d not in ([0] + routes[idx].nodes + [0])
            }

        return candidates

    @staticmethod
    def create_child_nodes(
        parent: "BranchNode",
        divergence_node: int,
        arc_set_1: List[Tuple[int, int]],
        arc_set_2: List[Tuple[int, int]],
        strength: float = 0.5,
    ) -> Tuple["BranchNode", "BranchNode"]:
        """
        Create two child nodes that forbid different arc sets.

        Args:
            parent: The node being branched.
            divergence_node: The node where divergence occurs.
            arc_set_1: First set of outgoing arcs to forbid.
            arc_set_2: Second set of outgoing arcs to forbid.

        Returns:
            (left_child, right_child)
        """
        from logic.src.policies.helpers.solvers_and_matheuristics.common.node import BranchNode

        # Child 1: Forbid arcs in arc_set_1
        constraints_1 = [EdgeBranchingConstraint(u, v, must_use=False) for u, v in arc_set_1]

        # Child 2: Forbid arcs in arc_set_2
        constraints_2 = [EdgeBranchingConstraint(u, v, must_use=False) for u, v in arc_set_2]

        hint = parent.lp_bound if parent.lp_bound is not None else 0.0

        # Paper §3.2: "Note that we first explore child node 2 (where arcs in
        # A(d, a₂) are forbidden) because path p is still allowed for commodity k."
        #
        # arc_set_1 holds the *higher-flow* half from the median split (path p's arc).
        # arc_set_2 holds the *lower-flow* complement.
        #
        # right_child forbids arc_set_2 → keeps arc_set_1 (path p) → paper's Child 2.
        # left_child  forbids arc_set_1 → loses path p              → paper's Child 1.
        #
        # DFS selects the node with the HIGHER lp_bound_hint first.  Giving right_child
        # (paper's preferred first-explore child) the unpenalised hint ensures it is
        # processed before left_child, reproducing the convergence behaviour reported
        # in Barnhart, Hane, and Vance (2000) Table 3/4.
        left_hint = hint - (strength * 1e-4)  # slight penalty — explored second

        left = BranchNode(
            constraints=constraints_1,  # type: ignore[arg-type]
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=left_hint,
            branching_rule="divergence",
        )
        right = BranchNode(
            constraints=constraints_2,  # type: ignore[arg-type]
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=hint,  # paper's preferred child — explored first
            branching_rule="divergence",
        )

        return left, right


class RyanFosterBranching:
    """
    Ryan-Foster branching strategy for Set Partitioning Problems.

    Identifies a node pair (r, s) that are "fractionally co-occurring" in
    the current LP solution and partitions the search space to eliminate
    this fractional state.

    Mathematical Basis:
    -------------------
    Based on Proposition 1 of Ryan and Foster (1981):
    If a Set Partitioning Problem has a fractional solution λ, there must
    exist two nodes r and s such that the set of routes visiting both
    r and s has a fractional sum:
        0 < ∑_{k: r ∈ route_k, s ∈ route_k} λ_k < 1

    Branching Rule:
    ---------------
    - Left Branch: r and s MUST be in the same route (∑_{k: r,s ∈ route_k} λ_k = 1).
      Any route visiting only one of the pair is disabled.
    - Right Branch: r and s MUST NOT be in the same route (∑_{k: r,s ∈ route_k} λ_k = 0).
      Any route visiting both nodes is disabled.

    Theoretical Rationale for VRPP:
    -------------------------------
    Ryan-Foster branching is polyhedrally stronger than simple edge-based
    branching for VRPs with identical vehicles (anonymous fleet). By
    constraining node pairs, the search space is partitioned without
    inducing the massive symmetry issues of arc-fixing. Furthermore, it is
    easily enforced in the RCSPP pricing subproblem by forbidding or
    requiring specific label transitions without increasing state space
    complexity.

    Reference: Ryan & Foster (1981), Proposition 1.
    """

    @staticmethod
    def find_branching_pair(
        routes: List[Route],
        route_values: Dict[int, float],
        mandatory_nodes: Set[int],
        tol: float = 1e-5,
    ) -> Optional[Tuple[Tuple[int, int], float]]:
        """
        Find the node pair (r, s) whose fractional co-occurrence is closest to 0.5.

        Ryan & Foster (1981) Proposition 1 guarantees that for any fractional SPP
        solution there exists such a pair.  Among all fractional pairs we select the
        one with ``|together_sum − round(together_sum)|`` maximised (i.e. closest to
        0.5), which produces the strongest symmetry-breaking branch (equal LP bound
        reduction on both children) and leads to fewer B&B nodes overall.

        The previous implementation returned the *first* fractional pair encountered
        (dependent on arbitrary dict-iteration order), which could produce very
        lopsided branches.  This version scores all candidates and returns the best.

        Args:
            routes: All routes in the master problem.
            route_values: Current LP solution {route_index: λ_k}.
            mandatory_nodes: Set of mandatory nodes needing exact partitioning.
            tol: Integrality tolerance.

        Returns:
            ((node_r, node_s), together_sum) to branch on, or None if integer.
        """
        best_pair: Optional[Tuple[int, int]] = None
        best_together_sum: float = 0.0
        best_frac: float = -1.0

        candidate_nodes: Set[int] = set()
        for idx, val in route_values.items():
            if abs(val - round(val)) > tol:
                candidate_nodes.update(routes[idx].node_coverage)

        candidate_list = sorted(candidate_nodes)

        for i, r in enumerate(candidate_list):
            for s in candidate_list[i + 1 :]:
                together_sum = sum(
                    v
                    for idx, v in route_values.items()
                    if r in routes[idx].node_coverage and s in routes[idx].node_coverage
                )
                frac = abs(together_sum - round(together_sum))
                if tol < frac < 1.0 - tol and frac > best_frac:
                    best_frac = frac
                    best_pair = (r, s)
                    best_together_sum = together_sum

        if best_pair is None:
            return None
        return best_pair, best_together_sum

    @staticmethod
    def create_child_nodes(
        parent: "BranchNode",
        node_r: int,
        node_s: int,
        together_sum: float = 0.5,
    ) -> Tuple["BranchNode", "BranchNode"]:
        """
        Create together (left) and separate (right) child nodes.

        Args:
            parent: The node being branched.
            node_r: First node in the pair.
            node_s: Second node in the pair.
            together_sum: Fractional co-occurrence value for tie-breaking.

        Returns:
            (left_child, right_child)
        """
        from logic.src.policies.helpers.solvers_and_matheuristics.common.node import BranchNode

        hint = parent.lp_bound if parent.lp_bound is not None else 0.0
        # Separation branch gets small penalty
        right_hint = hint - (together_sum * 1e-4)

        left = BranchNode(
            constraints=[RyanFosterBranchingConstraint(node_r, node_s, together=True)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=hint,
        )
        right = BranchNode(
            constraints=[RyanFosterBranchingConstraint(node_r, node_s, together=False)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=right_hint,
        )

        return left, right


class FleetSizeBranching:
    """Logic for branching on the total fleet size."""

    @staticmethod
    def find_fleet_branching(
        route_values: Dict[int, float],
        tol: float = 1e-4,
    ) -> Optional[float]:
        """Check if sum of lambdas is fractional."""
        fleet_usage = sum(route_values.values())
        if abs(fleet_usage - round(fleet_usage)) > tol:
            return fleet_usage
        return None

    @staticmethod
    def create_child_nodes(parent: "BranchNode", fleet_usage: float) -> Tuple["BranchNode", "BranchNode"]:
        """Create floor (lower branch) and ceiling (upper branch) child nodes."""
        from logic.src.policies.helpers.solvers_and_matheuristics.common.node import BranchNode

        floor = math.floor(fleet_usage + 1e-6)
        ceil = math.ceil(fleet_usage - 1e-6)
        lower_branch = BranchNode(
            constraints=[FleetSizeBranchingConstraint(floor, is_upper=True)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=parent.lp_bound,
        )
        upper_branch = BranchNode(
            constraints=[FleetSizeBranchingConstraint(ceil, is_upper=False)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=parent.lp_bound,
        )
        return lower_branch, upper_branch


class NodeVisitationBranching:
    """Logic for branching on optional node visitation."""

    @staticmethod
    def find_node_branching(
        routes: List[Route],
        route_values: Dict[int, float],
        optional_nodes: Set[int],
        tol: float = 1e-4,
    ) -> Optional[Tuple[int, float]]:
        """Find an optional node with fractional visitation."""
        for node in sorted(optional_nodes):
            visitation = sum(val for idx, val in route_values.items() if node in routes[idx].node_coverage)
            if tol < visitation < 1.0 - tol:
                return node, visitation
        return None

    @staticmethod
    def create_child_nodes(parent: "BranchNode", node: int, visitation: float) -> Tuple["BranchNode", "BranchNode"]:
        """Create v_i = 0 and v_i = 1 child nodes."""
        from logic.src.policies.helpers.solvers_and_matheuristics.common.node import BranchNode

        left = BranchNode(
            constraints=[NodeVisitationBranchingConstraint(node, forced=False)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=parent.lp_bound,
        )
        right = BranchNode(
            constraints=[NodeVisitationBranchingConstraint(node, forced=True)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=parent.lp_bound,
        )
        return left, right
