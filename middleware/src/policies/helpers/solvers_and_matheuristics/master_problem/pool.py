"""
Global Cut Pool for Branch-and-Price-and-Cut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Optional, Set, Tuple

if TYPE_CHECKING:
    from .problem_support import MasterProblemSupport


@dataclass
class CutInfo:
    """Metadata describing a generated valid inequality."""

    type: str  # e.g., 'rcc', 'sec', 'sri', 'lci'
    data: Any  # node-set, RHS, lifting coeffs, etc.
    active: bool = True
    violation: float = 0.0


class GlobalCutPool:
    """
    Centralized repository for globally valid inequalities across B&B nodes.

    Philosophy:
    In BPC, separation is expensive. By pooling valid inequalities (RCC, SRI, SEC 2.1)
    globally, we ensure that a cut discovered in one branch tightens the LP bound
    in sibling and child branches immediately, avoiding redundant separation and
    reducing the total number of B&B nodes explored.

    RCC storage note:
        RCC cuts are stored as (node_set, rhs) pairs so that the original RHS
        (= 2*⌈demand(S)/Q⌉, computed at discovery) is faithfully replayed when
        the cut is re-injected at descendant nodes. Storing only the node set
        and hard-coding rhs=1.0 would produce trivially weak cuts.

        Assumption: Customer demands and vehicle capacities are static throughout
        the B&B tree. If these were dynamic (e.g., stochastic demands handled at
        internal nodes), the RHS of purely node-set-based cuts could change,
        invalidating the global mathematical integrity of this archive.
    """

    def __init__(self) -> None:
        """Initialise empty global cut registries."""
        # RCC: maps node_set -> original rhs (2*ceil(demand/Q))
        self.rcc_cuts: Dict[FrozenSet[int], float] = {}
        self.sri_cuts: Set[FrozenSet[int]] = set()
        self.active_sri_vectors: Dict[FrozenSet[int], Dict[str, float]] = {}
        self.sec_cuts: Set[FrozenSet[int]] = set()  # Form 2.1 (Global)
        self.edge_clique_cuts: Set[Tuple[int, int]] = set()
        # LCI: maps node_set -> (rhs, route_coefficients, node_alphas)
        # node_alphas: per-node lifting coefficients for pricing (Barnhart et al. 2000 §4.2)
        self.lci_cuts: Dict[FrozenSet[int], Tuple[float, Dict[int, float], Dict[int, float]]] = {}
        # Optional source arc (i, j) for arc-saturation LCI (SaturatedArcLCIEngine).
        # When set, the pricing dual fires ONLY when the DP traverses that specific arc,
        # not on any visit to a node in the cover set.  None for node/capacity LCI.
        self.lci_arcs: Dict[FrozenSet[int], Optional[Tuple[int, int]]] = {}

    def add_cut(self, cut_type: str, data: Any) -> None:
        """
        Archive a globally valid cut in the pool.

        Global Validity Contract:
            Only archive cuts here that are valid at EVERY node in the B&B tree
            (e.g., RCC, SRI, SEC 2.1). Node-local cuts (branching-dependent SECs,
            lifted cover cuts dependent on local bounds) MUST NOT be added to
            this pool or they will lead to incorrect pruning and loss of optimality.

        Args:
            cut_type: The type of cut ("rcc", "sri", "sec_2.1", "edge_clique", "lci").
            data: Cut-specific data.
                  For "rcc": (FrozenSet[int], rhs: float)
                  For "lci": (FrozenSet[int], rhs: float, route_coefficients: Dict[int, float],
                              node_alphas: Dict[int, float])  — 4-tuple (node_alphas optional)
                  For others: FrozenSet[int] or Tuple[int, int].
        """

        if cut_type == "rcc":
            node_set, rhs = data
            # Only archive if better (tighter) than any existing cut on this set.
            existing = self.rcc_cuts.get(node_set, 0.0)
            if rhs > existing:
                self.rcc_cuts[node_set] = rhs
        elif cut_type == "sri":
            node_set, coeff_vec = data
            self.sri_cuts.add(node_set)
            self.active_sri_vectors[node_set] = coeff_vec
        elif cut_type == "sec_2.1":
            self.sec_cuts.add(data)
        elif cut_type == "edge_clique":
            self.edge_clique_cuts.add(data)
        elif cut_type == "lci":
            # Accept 3-tuple, 4-tuple (+ node_alphas), or 5-tuple (+ arc) variants.
            # 5-tuple: (node_set, rhs, coefficients, node_alphas, arc)
            # 4-tuple: (node_set, rhs, coefficients, node_alphas)
            # 3-tuple: (node_set, rhs, coefficients)
            if len(data) == 5:
                node_set, rhs, coefficients, node_alphas, arc = data
            elif len(data) == 4:
                node_set, rhs, coefficients, node_alphas = data
                arc = None
            else:
                node_set, rhs, coefficients = data
                node_alphas = {}
                arc = None
            self.lci_cuts[node_set] = (rhs, coefficients, node_alphas)
            self.lci_arcs[node_set] = arc

    def apply_to_master(self, master: MasterProblemSupport) -> int:
        """
        Inject all pooled global cuts into a fresh Master Problem instance.
        Typically called when entering a new B&B node to tighten the root relaxation.

        Args:
            master: The VRPPMasterProblem instance to receive the cuts.

        Returns:
            Number of cuts successfully applied.
        """
        added = 0
        # RCC: replay with the stored (correct) RHS, not a hard-coded value.
        for node_set, rhs in self.rcc_cuts.items():
            if master.add_capacity_cut(list(node_set), rhs=rhs, _skip_pool=True):
                added += 1
        for nodes in self.sri_cuts:
            if master.add_subset_row_cut(nodes):
                added += 1
        for nodes in self.sec_cuts:
            # Form 2.1 is always global
            if master.add_sec_cut(nodes, rhs=1.0, facet_form="2.1"):
                added += 1
        # Re-inject Edge Clique cuts from the global pool.
        for edge_tuple in self.edge_clique_cuts:
            if master.add_edge_clique_cut(edge_tuple[0], edge_tuple[1]):
                added += 1
        # Re-inject LCI cuts — recompute route coefficients from node_alphas using the
        # current master's route list.  The stale coefficients stored at discovery time
        # reference route indices from the discovery B&B node; at a descendant node the
        # master may have a different route pool, so those indices are meaningless.
        for node_set, lci_data in self.lci_cuts.items():
            if len(lci_data) == 3:
                rhs, _stale_coefficients, node_alphas = lci_data
            else:
                rhs, _stale_coefficients = lci_data  # type: ignore[misc]
                node_alphas = {}
            # Retrieve the source arc stored at discovery time (None for node-capacity LCI).
            lci_arc: Optional[Tuple[int, int]] = self.lci_arcs.get(node_set)
            new_coefficients: Dict[int, float] = {}
            for idx, route in enumerate(master.routes):  # type: ignore[union-attr]
                alpha_k = sum(node_alphas.get(n, 1.0 if n in node_set else 0.0) for n in route.node_coverage if n != 0)
                if alpha_k > 1e-6:
                    new_coefficients[idx] = alpha_k
            if master.add_lci_cut(list(node_set), rhs, new_coefficients, node_alphas=node_alphas, arc=lci_arc):
                added += 1
        return added
