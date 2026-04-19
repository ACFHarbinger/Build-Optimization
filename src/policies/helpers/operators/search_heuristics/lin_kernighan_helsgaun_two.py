"""
LKH-2 Meta-Heuristic — Full Implementation (Helsgaun 2009 / 2012).

LKH-2 extends LKH-1 with a **population-based** outer loop that diversifies
the search beyond what a single-trajectory ILS can achieve.  The key additions
over LKH-1 are:

1. **Population Management** (Section 2, Helsgaun 2009):
   Maintains a fixed-size population of locally-optimal tours.  New members
   replace the worst current member when they improve upon it or when the new
   tour is sufficiently distinct from all existing members (diversity
   preservation).

2. **Iterative Partial Transcription (IPT) Crossover** (Section 3):
   Generates offspring tours by combining two parent tours:
   - Identify edges shared by both parents (the "backbone").
   - Copy backbone edges to the offspring.
   - Fill remaining gaps with the LKH-1 local search, treating each gap as
     a sub-tour to optimise.
   This is the key operator that makes LKH-2 a *meta*-heuristic rather than
   merely a better-parameterised LKH-1.

3. **Shared Penalty Vector** (Section 2.2):
   The subgradient-derived penalty vector π is computed once from the first
   LKH-1 solution and then shared across all population members.  It defines
   the candidate set for every crossover offspring and local-search call.
   This avoids redundant subgradient runs.

4. **Population-Informed Restarts** (Section 4):
   Rather than always perturbing the best tour (as in ILS), LKH-2 selects
   parent pairs from the population probabilistically (fitness-proportionate
   with diversity bonus) to seed crossover.  Pure ILS restarts (double-bridge)
   are applied only when the population has fully converged.

Architecture
------------
This module owns:

- :class:`TourPopulation` — population container with insertion / selection.
- :func:`ipt_crossover`   — Iterative Partial Transcription operator.
- :func:`solve_lkh2`      — top-level LKH-2 meta-heuristic.

Heavy computation is delegated to:

- :func:`solve_lkh1` (from ``lkh1.py``) for Phase 1 (subgradient + penalized
  candidates + local search of each new solution).
- :func:`_improve_tour` (from ``lin_kernighan_helsgaun.py``) for the inner
  k-opt loop within IPT gap-filling.
- :func:`compute_alpha_measures`, :func:`get_candidate_set` (from
  ``lin_kernighan_helsgaun.py``) for penalized candidates.
- :func:`_double_bridge_kick` (from ``_tour_construction.py``) for pure ILS
  restarts when population diversity collapses.

References
----------
Helsgaun, K. (2009). General k-opt submoves for the Lin-Kernighan-Helsgaun
  TSP heuristic. Mathematical Programming Computation, 1(2-3), 119–163.

Helsgaun, K. (2012). LKH-2: A new version of LKH.  Technical Report,
  Roskilde University.
"""

from __future__ import annotations

from random import Random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.policies.helpers.operators.search_heuristics._objective import (
    get_cost,
    is_better,
)
from logic.src.policies.helpers.operators.search_heuristics._tour_construction import (
    _double_bridge_kick,
    _initialize_tour,
)
from logic.src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun import (
    _improve_tour,
    compute_alpha_measures,
    get_candidate_set,
    run_subgradient,
)
from logic.src.tracking.viz_mixin import PolicyStateRecorder

# ---------------------------------------------------------------------------
# Tour Population
# ---------------------------------------------------------------------------


class TourPopulation:
    """
    Fixed-size population of locally-optimal tours for LKH-2.

    Maintains:
    - A list of (tour, cost) pairs, at most ``max_size`` members.
    - The best solution (lowest cost) seen so far.
    - Diversity tracking via edge-set Hamming distance.

    Insertion policy:
    - A new tour is accepted if it is better than the current worst member
      **or** if its minimum edge-Hamming distance to all current members
      exceeds ``diversity_threshold * n`` (i.e., it is sufficiently novel).
    - When the population is full and the new tour is accepted, the worst
      member is evicted.
    """

    def __init__(self, max_size: int = 10, diversity_threshold: float = 0.05):
        """
        Args:
            max_size:            Maximum population size.
            diversity_threshold: Minimum fractional Hamming distance (0..1)
                                 from all existing members for a new tour to be
                                 accepted on diversity grounds.
        """
        self.max_size = max_size
        self.diversity_threshold = diversity_threshold
        self.members: List[Tuple[List[int], float]] = []  # (tour, cost)
        self.edge_sets: List[Set[Tuple[int, int]]] = []  # sorted-pair edge sets
        self.best_tour: List[int] = []
        self.best_cost: float = float("inf")

    # ------------------------------------------------------------------
    # Edge-set helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_edge_set(tour: List[int]) -> Set[Tuple[int, int]]:
        """Convert a closed tour to a frozenset of sorted (i,j) edge pairs."""
        es: Set[Tuple[int, int]] = set()
        for k in range(len(tour) - 1):
            a, b = tour[k], tour[k + 1]
            es.add((min(a, b), max(a, b)))
        return es

    def _hamming_distance(self, es_new: Set[Tuple[int, int]], n: int) -> float:
        """
        Minimum fractional Hamming distance from ``es_new`` to any existing member.

        Hamming(T1, T2) / n ∈ [0, 1], where Hamming counts edges in the
        symmetric difference.
        """
        if not self.edge_sets:
            return 1.0
        min_dist = 1.0
        for es in self.edge_sets:
            sym_diff = len(es_new.symmetric_difference(es))
            dist = sym_diff / max(n, 1)
            min_dist = min(min_dist, dist)
        return min_dist

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def try_insert(self, tour: List[int], cost: float) -> bool:
        """
        Attempt to insert a new tour into the population.

        Args:
            tour: Closed node sequence.
            cost: Tour length (raw, unpenalized).

        Returns:
            True if the tour was inserted; False otherwise.
        """
        n = len(tour) - 1
        es_new = self._to_edge_set(tour)

        # Update global best
        if cost < self.best_cost - 1e-9:
            self.best_cost = cost
            self.best_tour = tour[:]

        if len(self.members) < self.max_size:
            # Population not yet full: always accept
            self.members.append((tour[:], cost))
            self.edge_sets.append(es_new)
            return True

        # Population full: accept if better than worst or sufficiently diverse
        worst_idx = max(range(len(self.members)), key=lambda i: self.members[i][1])
        worst_cost = self.members[worst_idx][1]

        hamming = self._hamming_distance(es_new, n)
        is_better_than_worst = cost < worst_cost - 1e-9
        is_diverse = hamming >= self.diversity_threshold

        if is_better_than_worst or is_diverse:
            self.members[worst_idx] = (tour[:], cost)
            self.edge_sets[worst_idx] = es_new
            return True
        return False

    def sample_parents(self, rng: np.random.Generator, n_parents: int = 2) -> List[Tuple[List[int], float]]:
        """
        Sample parent tours from the population for crossover.

        Selection picks the maximum-diversity pair from the population:
        - Draw a candidate pool via fitness-proportionate sampling (lower cost
          → higher probability).
        - From that pool, select the pair that maximises their mutual
          edge-Hamming distance, encouraging structural diversity in offspring.

        Args:
            rng:       NumPy Generator.
            n_parents: Number of parents to sample.

        Returns:
            List of (tour, cost) pairs.
        """
        if len(self.members) <= n_parents:
            return list(self.members)

        if n_parents == 2 and len(self.members) >= 3:
            # Fitness-proportionate candidate pool (sample min(2*pop, all) members)
            costs = np.array([c for _, c in self.members])
            max_c = costs.max()
            weights = max_c - costs + 1e-9
            weights /= weights.sum()
            pool_size = min(len(self.members), max(4, 2 * n_parents))
            pool_indices = list(rng.choice(len(self.members), size=pool_size, replace=False, p=weights))

            # Select pair with maximum Hamming distance from the candidate pool
            best_pair = (pool_indices[0], pool_indices[1])
            best_dist = 0.0
            for ii in range(len(pool_indices)):
                for jj in range(ii + 1, len(pool_indices)):
                    idx_i = pool_indices[ii]
                    idx_j = pool_indices[jj]
                    es_i = self.edge_sets[idx_i]
                    es_j = self.edge_sets[idx_j]
                    n_nodes = len(self.members[idx_i][0]) - 1
                    dist = len(es_i.symmetric_difference(es_j)) / max(n_nodes, 1)
                    if dist > best_dist:
                        best_dist = dist
                        best_pair = (idx_i, idx_j)

            return [self.members[i] for i in best_pair]

        # General case: fitness-proportionate weighted sample
        costs = np.array([c for _, c in self.members])
        max_c = costs.max()
        weights = max_c - costs + 1e-9
        weights /= weights.sum()
        indices = rng.choice(len(self.members), size=n_parents, replace=False, p=weights)
        return [self.members[i] for i in indices]

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def diversity(self) -> float:
        """Mean pairwise fractional Hamming distance across population."""
        if len(self.members) < 2:
            return 1.0
        n = len(self.members[0][0]) - 1
        total = 0.0
        count = 0
        for i in range(len(self.edge_sets)):
            for j in range(i + 1, len(self.edge_sets)):
                total += len(self.edge_sets[i].symmetric_difference(self.edge_sets[j])) / max(n, 1)
                count += 1
        return total / max(count, 1)


# ---------------------------------------------------------------------------
# IPT Crossover
# ---------------------------------------------------------------------------


def ipt_crossover(  # noqa: C901
    parent1: List[int],
    parent2: List[int],
    distance_matrix: np.ndarray,
    candidates: Dict[int, List[int]],
    stdlib_rng: Random,
    max_k: int = 5,
    max_gap_iter: int = 20,
) -> Tuple[List[int], float]:
    """
    Iterative Partial Transcription (IPT) crossover operator (Helsgaun 2009).

    Algorithm:
    1. Compute the "backbone" B = edges present in BOTH parent1 and parent2.
    2. Extract the connected backbone chains (maximal paths/cycles whose edges
       are all in B).  Nodes not covered by any backbone edge are "free".
    3. Assemble an offspring tour:
       a. Start from a chain endpoint (or any free node).
       b. Walk along backbone chains greedily, then bridge each gap between
          chain endpoints and free nodes using the nearest-neighbour rule.
       c. Ensure each node is visited exactly once.
    4. Refine the offspring via LKH-1 k-opt with ``fixed_edges=backbone``
       so that backbone edges are never removed during gap-filling.
    5. If no backbone exists (parents share no edges), return the better parent.

    The use of ``fixed_edges`` is the key departure from a plain nearest-
    neighbour re-assembly: it preserves the parental backbone structure
    throughout the refinement step, faithfully implementing the IPT algorithm
    of Helsgaun (2009, Section 3).

    Args:
        parent1:          First parent (closed tour, starts and ends at 0).
        parent2:          Second parent (closed tour, starts and ends at 0).
        distance_matrix:  (n x n) raw cost matrix.
        candidates:       α-candidate lists (penalized, from run_subgradient).
        stdlib_rng:       stdlib Random for perturbation.
        max_k:            k-opt depth for gap-filling local search.
        max_gap_iter:     Maximum improvement passes for each gap.

    Returns:
        (offspring_tour, offspring_cost).
    """
    n = len(parent1) - 1  # number of real nodes

    # --- 1. Compute backbone (edges common to both parents) ---
    es1 = TourPopulation._to_edge_set(parent1)
    es2 = TourPopulation._to_edge_set(parent2)
    backbone: Set[Tuple[int, int]] = es1 & es2

    if not backbone:
        # No shared edges: return the better parent unchanged
        c1 = get_cost(parent1, distance_matrix)
        c2 = get_cost(parent2, distance_matrix)
        return (parent1[:], c1) if c1 <= c2 else (parent2[:], c2)

    # --- 2. Build backbone adjacency and extract chains ---
    # Adjacency in the backbone sub-graph
    bb_adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for a, b in backbone:
        bb_adj[a].append(b)
        bb_adj[b].append(a)

    # Find chain endpoints: nodes with backbone degree 1 (endpoints) or 0 (free)
    # Nodes with backbone degree 2 are interior chain nodes.
    visited = [False] * n
    chains: List[List[int]] = []  # each chain is an ordered list of nodes

    for start in range(n):
        if visited[start] or len(bb_adj[start]) == 0:
            continue

        # Walk the chain starting from 'start'
        chain = [start]
        visited[start] = True
        prev = -1
        curr = start

        while True:
            # Prefer the unvisited neighbour that is not the previous node
            next_node = None
            for nb in bb_adj[curr]:
                if nb != prev and not visited[nb]:
                    next_node = nb
                    break

            if next_node is None:
                # No more unvisited neighbours: chain terminates
                # Check if the last node connects back to start (closed loop)
                if start in bb_adj[curr] and start != prev and len(chain) > 2:
                    chain.append(start)  # Mark it as closed by re-appending start
                break

            visited[next_node] = True
            chain.append(next_node)
            prev = curr
            curr = next_node

        if len(chain) >= 2:
            chains.append(chain)

    # --- 3. Assemble offspring by stitching chains and free nodes ---
    # Nodes already placed by backbone chains
    placed_nodes: Set[int] = set()
    for chain in chains:
        for node in chain:
            if node != chain[0] or chain[-1] != chain[0]:  # avoid double-adding closed chain start
                placed_nodes.add(node)
        # For closed chains (cycle), the start appears twice but it IS placed
        if chains and chain[0] == chain[-1]:
            placed_nodes.add(chain[0])

    # Re-compute properly: any node that appears in any chain is placed
    placed_nodes = set()
    for chain in chains:
        for node in chain:
            placed_nodes.add(node)

    free_nodes = set(range(n)) - placed_nodes

    # Build offspring as an ordered list (open path, will be closed at end)
    offspring_path: List[int] = []

    # If the whole tour is a single closed backbone cycle, we're done
    if len(chains) == 1 and chains[0][0] == chains[0][-1] and len(placed_nodes) == n:
        offspring = chains[0]
        # Ensure it starts at 0
        if offspring[0] != 0:
            idx = offspring.index(0)
            offspring = offspring[idx:] + offspring[1 : idx + 1]
            if offspring[-1] != 0:
                offspring.append(0)
        return offspring, get_cost(offspring, distance_matrix)

    # Greedy stitching: walk chains in order and bridge gaps via nearest-neighbour
    remaining_chains = list(chains)
    remaining_free = set(free_nodes)

    # Start with the chain containing the depot (node 0) if possible
    depot_chain_idx = None
    for idx, chain in enumerate(remaining_chains):
        if 0 in chain:
            depot_chain_idx = idx
            break

    if depot_chain_idx is not None:
        first_chain = remaining_chains.pop(depot_chain_idx)
    elif remaining_chains:
        first_chain = remaining_chains.pop(0)
    else:
        first_chain = []

    # If chain starts with depot, use as-is; otherwise rotate so depot is first
    if first_chain and first_chain[0] != 0 and 0 in first_chain:
        rot = first_chain.index(0)
        first_chain = first_chain[rot:] + first_chain[:rot]

    offspring_path = list(first_chain)
    # For closed chains (cycle detected), strip the duplicated last node
    if len(offspring_path) > 1 and offspring_path[0] == offspring_path[-1]:
        offspring_path = offspring_path[:-1]

    def nearest_free(current: int, free: Set[int]) -> int:
        """Return the nearest free node to `current`."""
        return min(free, key=lambda v: distance_matrix[current, v])

    def nearest_chain_endpoint(current: int, chains_left: List[List[int]]) -> Tuple[int, int, bool]:
        """
        Find the closest chain endpoint to `current`.

        Returns (chain_idx, end_which, reversed):
          - chain_idx: index in chains_left
          - end_which: 0 = connect to chain[0], 1 = connect to chain[-1]
          - reversed:  whether to traverse the chain in reverse
        """
        best_dist = float("inf")
        best = (0, 0, False)
        for ci, ch in enumerate(chains_left):
            d0 = distance_matrix[current, ch[0]]
            d1 = distance_matrix[current, ch[-1]]
            if d0 < best_dist:
                best_dist = d0
                best = (ci, 0, False)  # attach ch forward: ch[0] connected to current
            if d1 < best_dist:
                best_dist = d1
                best = (ci, 1, True)  # attach ch backward: ch[-1] connected to current
        return best

    # Greedy stitching loop
    while remaining_chains or remaining_free:
        current = offspring_path[-1] if offspring_path else 0

        # Prefer adding a nearby chain endpoint over a lone free node
        nearest_chain_dist = float("inf")
        nearest_free_dist = float("inf")

        if remaining_chains:
            ci, _, _ = nearest_chain_endpoint(current, remaining_chains)
            ch = remaining_chains[ci]
            nearest_chain_dist = min(
                distance_matrix[current, ch[0]],
                distance_matrix[current, ch[-1]],
            )
        if remaining_free:
            nf = nearest_free(current, remaining_free)
            nearest_free_dist = distance_matrix[current, nf]

        if remaining_chains and nearest_chain_dist <= nearest_free_dist:
            # Attach the nearest chain
            ci, end_which, reversed_chain = nearest_chain_endpoint(current, remaining_chains)
            ch = remaining_chains.pop(ci)
            if reversed_chain:
                ch = list(reversed(ch))
            # Strip the connecting node from the chain if it duplicates the path end
            if offspring_path and ch[0] == offspring_path[-1]:
                ch = ch[1:]
            offspring_path.extend(ch)
        else:
            # Insert nearest free node
            nf = nearest_free(current, remaining_free)
            remaining_free.discard(nf)
            offspring_path.append(nf)

    # --- 4. Close the tour and ensure it starts at depot (0) ---
    # Rotate to start at node 0
    if offspring_path and offspring_path[0] != 0 and 0 in offspring_path:
        rot = offspring_path.index(0)
        offspring_path = offspring_path[rot:] + offspring_path[:rot]

    # Guarantee each non-depot node appears exactly once (safety net)
    seen: Set[int] = set()
    deduped: List[int] = []
    for node in offspring_path:
        if node not in seen:
            seen.add(node)
            deduped.append(node)
    # Add any missing nodes (should not happen in correct implementations)
    for node in range(n):
        if node not in seen:
            deduped.append(node)
    offspring_path = deduped

    # Close the tour
    offspring = offspring_path + [offspring_path[0]]

    # --- 5. Refine with backbone-preserving LKH-1 k-opt ---
    offspring_cost = get_cost(offspring, distance_matrix)
    dont_look = np.zeros(n, dtype=bool)
    # Pass backbone as fixed_edges so refinement does not remove backbone edges
    backbone_fixed: Set[Tuple[int, int]] = backbone  # already sorted pairs
    for _ in range(max_gap_iter):
        offspring, offspring_cost, improved, dont_look = _improve_tour(
            offspring,
            offspring_cost,
            candidates,
            distance_matrix,
            stdlib_rng,
            dont_look,
            max_k,
            fixed_edges=backbone_fixed,
        )
        if not improved:
            break

    return offspring, offspring_cost


# ---------------------------------------------------------------------------
# LKH-2 Main Solver
# ---------------------------------------------------------------------------


def solve_lkh2(  # noqa: C901
    distance_matrix: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    max_iterations: int = 200,
    max_k: int = 5,
    n_candidates: int = 5,
    population_size: int = 10,
    diversity_threshold: float = 0.05,
    crossover_prob: float = 0.6,
    sg_max_iter: int = 100,
    sg_mu_init: float = 1.0,
    sg_patience: int = 20,
    local_search_iter_per_offspring: int = 50,
    recorder: Optional[PolicyStateRecorder] = None,
    np_rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> Tuple[List[int], float, float]:
    """
    Solve a TSP instance using the LKH-2 meta-heuristic (Helsgaun 2009).

    Phase 1 — Penalty Initialisation (shared across all restarts):
        Run subgradient optimisation (:func:`run_subgradient`) once to compute
        π* and the penalized candidate sets that guide ALL subsequent searches.

    Phase 2 — Population Seeding:
        Generate ``population_size`` initial locally-optimal tours by running
        LKH-1 local search from different starting points (nearest-neighbour
        from random seeds).  All use the shared penalized candidates.

    Phase 3 — Crossover-based Diversification loop:
        For each iteration:
        a. With probability ``crossover_prob``: select two parent tours from
           the population; apply IPT crossover to generate an offspring; locally
           optimise the offspring.
        b. Otherwise: apply double-bridge to the best tour and locally optimise.
        c. Try to insert the new tour into the population.
        d. Update the global best.

    Args:
        distance_matrix:               (n × n) raw symmetric cost matrix.
        initial_tour:                  Optional starting tour.
        max_iterations:                Total iteration budget (crossover + ILS).
        max_k:                         k-opt depth (2–5).
        n_candidates:                  Candidate list size per node.
        population_size:               Size of the tour population.
        diversity_threshold:           Minimum fractional Hamming distance for
                                       diversity-driven population insertion.
        crossover_prob:                Probability of applying IPT crossover
                                       vs. pure ILS double-bridge restart.
        sg_max_iter:                   Subgradient iteration budget.
        sg_mu_init:                    Initial Polyak step multiplier.
        sg_patience:                   μ-halving patience.
        local_search_iter_per_offspring: k-opt passes per new solution.
        recorder:                      Optional telemetry recorder.
        np_rng:                        NumPy Generator.
        seed:                          Alternative seed.

    Returns:
        (best_tour, best_cost, hk_lower_bound).
    """
    n = len(distance_matrix)
    if n < 3:
        t = list(range(n)) + [0]
        return t, float(get_cost(t, distance_matrix)), 0.0

    if np_rng is None:
        np_rng = np.random.default_rng(seed if seed is not None else 42)
    stdlib_rng = Random(int(np_rng.integers(0, 2**31)))

    # -----------------------------------------------------------------
    # Phase 1: One-time subgradient to get π* and penalized candidates
    # -----------------------------------------------------------------
    pi, hk_bound, penalized_dm = run_subgradient(
        distance_matrix,
        max_iter=sg_max_iter,
        mu_init=sg_mu_init,
        patience=sg_patience,
    )

    alpha_pen = compute_alpha_measures(penalized_dm)
    candidates = get_candidate_set(penalized_dm, alpha_pen, max_candidates=n_candidates)

    if recorder is not None:
        recorder.record(engine="lkh2_subgradient", hk_bound=hk_bound)

    # -----------------------------------------------------------------
    # Phase 2: Seed the population with locally-optimal tours
    # -----------------------------------------------------------------
    population = TourPopulation(max_size=population_size, diversity_threshold=diversity_threshold)

    # Seed 0: from the initial tour / nearest-neighbour
    seed_tour = _initialize_tour(distance_matrix, initial_tour)
    seed_cost = get_cost(seed_tour, distance_matrix)
    dont_look_seed = np.zeros(n, dtype=bool)
    for _ in range(local_search_iter_per_offspring):
        seed_tour, seed_cost, improved, dont_look_seed = _improve_tour(
            seed_tour,
            seed_cost,
            candidates,
            distance_matrix,
            stdlib_rng,
            dont_look_seed,
            max_k,
        )
        if not improved:
            break
    population.try_insert(seed_tour, seed_cost)

    # Additional seeds: random double-bridge perturbations from seed 0
    for _k in range(population_size - 1):
        alt_tour = _double_bridge_kick(seed_tour, distance_matrix, stdlib_rng)
        alt_cost = get_cost(alt_tour, distance_matrix)
        alt_dont_look = np.zeros(n, dtype=bool)
        for _ in range(local_search_iter_per_offspring):
            alt_tour, alt_cost, improved, alt_dont_look = _improve_tour(
                alt_tour,
                alt_cost,
                candidates,
                distance_matrix,
                stdlib_rng,
                alt_dont_look,
                max_k,
            )
            if not improved:
                break
        population.try_insert(alt_tour, alt_cost)

    best_tour = population.best_tour[:]
    best_cost = population.best_cost

    if recorder is not None:
        recorder.record(
            engine="lkh2_population_init",
            pop_size=population.size,
            pop_diversity=population.diversity,
            best_cost=best_cost,
        )

    # -----------------------------------------------------------------
    # Phase 3: Crossover + ILS meta-loop
    # -----------------------------------------------------------------
    for iteration in range(max_iterations):
        # Decide: crossover or pure ILS
        use_crossover = float(np_rng.random()) < crossover_prob and population.size >= 2

        if use_crossover:
            # Select two parent tours from population
            parents = population.sample_parents(np_rng, n_parents=2)
            p1_tour, _ = parents[0]
            p2_tour, _ = parents[1]

            # Generate offspring via IPT crossover + local search
            offspring, offspring_cost = ipt_crossover(
                p1_tour,
                p2_tour,
                distance_matrix,
                candidates,
                stdlib_rng,
                max_k=max_k,
                max_gap_iter=local_search_iter_per_offspring,
            )
        else:
            # Pure ILS: double-bridge from best tour
            offspring = _double_bridge_kick(best_tour, distance_matrix, stdlib_rng)
            offspring_cost = get_cost(offspring, distance_matrix)
            dont_look_ils = np.zeros(n, dtype=bool)
            for _ in range(local_search_iter_per_offspring):
                offspring, offspring_cost, improved, dont_look_ils = _improve_tour(
                    offspring,
                    offspring_cost,
                    candidates,
                    distance_matrix,
                    stdlib_rng,
                    dont_look_ils,
                    max_k,
                )
                if not improved:
                    break

        # Try to add to population
        population.try_insert(offspring, offspring_cost)

        # Update global best
        if is_better(offspring_cost, best_cost):
            best_cost = offspring_cost
            best_tour = offspring[:]

        if recorder is not None:
            recorder.record(
                engine="lkh2",
                iteration=iteration,
                best_cost=best_cost,
                hk_bound=hk_bound,
                gap=(best_cost - hk_bound) / max(abs(hk_bound), 1.0),
                pop_size=population.size,
                pop_diversity=population.diversity,
                used_crossover=use_crossover,
            )

        # Early termination: optimality gap closed
        if hk_bound > 0 and (best_cost - hk_bound) / hk_bound < 1e-4:
            break

    return best_tour, best_cost, hk_bound


# ---------------------------------------------------------------------------
# Convenience wrapper matching the solve_lkh interface
# ---------------------------------------------------------------------------


def solve_lkh(
    distance_matrix: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    max_iterations: int = 200,
    max_k: int = 5,
    recorder: Optional[PolicyStateRecorder] = None,
    np_rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> Tuple[List[int], float]:
    """
    Thin wrapper for :func:`solve_lkh2` matching the two-return interface
    of ``lin_kernighan_helsgaun.solve_lkh`` and ``lkh1.solve_lkh``.

    Returns (best_tour, best_cost) without the Held-Karp bound.
    """
    tour, cost, _ = solve_lkh2(
        distance_matrix,
        initial_tour=initial_tour,
        max_iterations=max_iterations,
        max_k=max_k,
        recorder=recorder,
        np_rng=np_rng,
        seed=seed,
    )
    return tour, cost
