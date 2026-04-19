"""
Genetic Programming Hyper-Heuristic (GPHH) — Constructive Heuristic Generator.

Instead of selecting between pre-built operators (heuristic selection), this
implementation *generates* a construction heuristic by evolving a GP scoring
function (heuristic generation, Burke et al., 2009).

**Constructive Algorithm**:

The GP tree acts as a learned priority function.  At each construction step:

  1. For each route, restrict candidates to the ``candidate_list_size``-nearest
     neighbours of the route's first and last nodes (K-NN candidate list).
     This reduces evaluation cost from O(N×R) to O(K×R) per step.
  2. Build an insertion context (4 local features) for each feasible candidate.
  3. Score with ``tree.evaluate(context)``.
  4. Execute the highest-scored insertion.
  5. Repeat until no capacity-feasible candidate yields a positive score.

**Scaling**:

Without candidate list: O(N² × R) tree evaluations per construction (N steps,
N nodes per step, R routes).  With K-NN list: O(N × K × R) — a factor of N/K
reduction.  For N=100, K=10 this is 10× faster.

**True Generalization** (Train/Test Paradigm):

Tree fitness is computed across fully distinct training instances, each with
its own distance matrix.  The adapter layer provides these via
``training_environments``.  If none are supplied, a node-subset fallback is
used on the test instance.

Reference:
    Burke, E. K., Hyde, M. R., Kendall, G., Ochoa, G., Ozcan, E., & Woodward, J. R.
    "Exploring Hyper-heuristic Methodologies with Genetic Programming", 2009
"""

import random
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .params import GPHHParams
from .tree import GPNode, _mutate, _random_tree, _subtree_crossover, to_callable

# Type alias for a training environment triple
TrainingEnv = Tuple[np.ndarray, Dict[int, float], List[int]]

# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


class GPHHSolver:
    """
    Genetic Programming Hyper-Heuristic solver for VRPP.

    Evolves a GP tree that scores candidate insertions during constructive
    solution building.  The best tree is applied to the full problem instance
    to produce the final routing solution.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: GPHHParams,
        mandatory_nodes: Optional[List[int]] = None,
        training_environments: Optional[List[TrainingEnv]] = None,
    ):
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes: Set[int] = set(mandatory_nodes or [])
        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))
        self.rng = random.Random(params.seed) if params.seed is not None else random.Random()
        self.training_environments = training_environments  # May be None

        # Precompute K-NN index for the test instance (used in final application)
        self._knn = self._build_knn(dist_matrix, self.nodes, params.candidate_list_size)

    # ------------------------------------------------------------------
    # K-NN candidate list (precomputed)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_knn(
        dm: np.ndarray,
        nodes: List[int],
        k: int,
    ) -> Dict[int, List[int]]:
        """
        Precompute the K-nearest neighbours for every node (including depot 0).

        For each node ``n``, the KNN list contains the ``k`` nodes closest to
        ``n`` by distance matrix lookup, excluding ``n`` itself.  The depot (0)
        is included as the anchor for opening new routes.

        Precomputation is O(N² log K).  Per-step lookup is O(1) (list index).

        Args:
            dm: Distance matrix (square, 0-indexed, row/col 0 = depot).
            nodes: Customer node indices.
            k: Number of nearest neighbours to retain per node.

        Returns:
            Mapping node → sorted list of up to ``k`` nearest nodes (by distance).
        """
        all_nodes = [0] + nodes  # Include depot
        knn: Dict[int, List[int]] = {}
        for n in all_nodes:
            # Sort every other node by distance, take the K closest
            others = [m for m in all_nodes if m != n]
            others.sort(key=lambda m: dm[n][m])
            knn[n] = others[:k]
        return knn

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run GPHH: evolve constructive insertion heuristics, apply best.

        1. Resolve training environments (external or fallback node-subset).
        2. Evolve GP population using averaged normalised fitness.
        3. Apply the best-evolved tree to the full instance.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()

        # Resolve training environments
        training = self._resolve_training()

        # Initialise GP population
        gp_pop = [_random_tree(self.params.tree_depth, self.rng) for _ in range(self.params.gp_pop_size)]
        gp_fitness = [self._evaluate_tree(tree, training) for tree in gp_pop]

        best_idx = int(np.argmax(gp_fitness))
        best_tree = gp_pop[best_idx].copy()
        best_fitness = gp_fitness[best_idx]

        # GP evolution loop
        for _gen in range(self.params.max_gp_generations):
            # Reserve 40% of time budget for final construction
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit * 0.6:
                break

            # --- Elitism: Preserve the best individuals ---
            # Sort population by fitness (descending index)
            ranked_indices = np.argsort(gp_fitness)[::-1]
            new_pop: List[GPNode] = []
            new_fitness: List[float] = []

            # Carry over the top 2 elites (at most)
            n_elites = min(2, self.params.gp_pop_size)
            for i in range(n_elites):
                idx = ranked_indices[i]
                new_pop.append(gp_pop[idx].copy())
                new_fitness.append(gp_fitness[idx])

            # --- Fill the remainder via Mutually Exclusive Operators ---
            # 80% Crossover, 20% Mutation as per standard GP flow (Koza).
            while len(new_pop) < self.params.gp_pop_size:
                roll = self.rng.random()

                if roll < 0.8:
                    # Crossover (produces 2 offspring)
                    p1 = self._tournament(gp_pop, gp_fitness)
                    p2 = self._tournament(gp_pop, gp_fitness)
                    offspring = list(_subtree_crossover(p1.copy(), p2.copy(), self.rng, self.params.tree_depth))
                else:
                    # Mutation (produces 1 offspring)
                    p = self._tournament(gp_pop, gp_fitness)
                    offspring = [_mutate(p.copy(), self.params.tree_depth, self.rng, self.params.tree_depth)]

                for tree in offspring:
                    if len(new_pop) < self.params.gp_pop_size:
                        f = self._evaluate_tree(tree, training)
                        new_pop.append(tree)
                        new_fitness.append(f)

                        if f > best_fitness:
                            best_tree = tree.copy()
                            best_fitness = f

            gp_pop = new_pop[: self.params.gp_pop_size]
            gp_fitness = new_fitness[: self.params.gp_pop_size]

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=_gen,
                best_tree_fitness=best_fitness,
                gp_pop_size=self.params.gp_pop_size,
            )

        # Apply best tree to the full (test) instance using its precomputed KNN
        best_routes = self._construct_solution(
            best_tree,
            self.nodes,
            self.wastes,
            self.mandatory_nodes,
            self.dist_matrix,
            self._knn,
        )

        profit = self._evaluate_routes(best_routes, self.wastes, self.dist_matrix)
        cost = self._cost(best_routes, self.dist_matrix)

        return best_routes, profit, cost

    # ------------------------------------------------------------------
    # Constructive solution building
    # ------------------------------------------------------------------

    def _construct_solution(
        self,
        tree: GPNode,
        nodes: List[int],
        wastes: Dict[int, float],
        mandatory: Set[int],
        dm: np.ndarray,
        knn: Dict[int, List[int]],
    ) -> List[List[int]]:
        """
        Build a VRPP solution by greedily inserting nodes scored by the GP tree.

        **K-NN Candidate Filtering** (scaling fix):

        For each existing route, instead of evaluating all N unvisited nodes,
        only the K-nearest neighbours of the route's *first and last* nodes
        are considered as candidates.  This reduces the inner loop from O(N)
        to O(K) candidates per route, giving O(N × K × R) total evaluations
        instead of O(N² × R).

        For opening a new route, the K-nearest neighbours of the depot (node 0)
        are used as candidates, since the depot-return cost dominates.

        **No Profitability Gate**:

        All capacity-feasible candidates are passed to the GP tree for scoring.
        The tree must learn to assign heavily negative scores to unprofitable
        insertions.  This allows the GP to discover strategies that make
        temporarily unprofitable moves to access profitable spatial clusters.

        The loop terminates when no candidate produces a score above
        ``float("-inf")`` (i.e., no feasible candidate exists).  Mandatory nodes
        are inserted even when they would otherwise have no scored candidate by
        falling back to a depot-anchored forced insertion.

        Args:
            tree: GP scoring function.
            nodes: Customer node indices to consider.
            wastes: Waste/value of each node.
            mandatory: Nodes that must be visited regardless.
            dm: Distance matrix for this instance.
            knn: K-NN index keyed by node index (includes depot 0).

        Returns:
            Constructed routes (list of node-index lists).
        """
        routes: List[List[int]] = []
        route_loads: List[float] = []  # Cached loads for O(1) capacity checks
        visited: Set[int] = set()
        cap = self.capacity

        # Pre-index nodes as a set for O(1) membership checks
        node_set: Set[int] = set(nodes)

        # Compile the GP tree once at the start of construction for speed
        func = to_callable(tree)
        while True:
            best_score = -float("inf")
            best_action: Optional[Tuple[int, int, int]] = None  # (node, route_idx, pos)

            # ------------------------------------------------------------------
            # Score candidates for each existing route
            # ------------------------------------------------------------------
            for ri, route in enumerate(routes):
                # Expanded K-NN search: evaluate neighbours of EVERY node in the
                # current route to prevent spatial blindspots.
                candidates = set()
                for node_in_route in route:
                    candidates.update(knn.get(node_in_route, []))
                candidates = (candidates & node_set) - visited

                for node in candidates:
                    node_waste = wastes.get(node, 0.0)
                    if route_loads[ri] + node_waste > cap:
                        continue  # Capacity violation — skip

                    pos, delta = self._cheapest_insertion(route, node, dm)
                    node_revenue = node_waste * self.R

                    # Direct call to compiled lambda (bypasses dict construction and recursion)
                    score = func(
                        node_profit=node_revenue,
                        distance_to_route=self._min_distance_to_route(node, route, dm),
                        insertion_cost=delta,
                        remaining_capacity=cap - route_loads[ri] - node_waste,
                    )

                    if score > best_score:
                        best_score = score
                        best_action = (node, ri, pos)

            # ------------------------------------------------------------------
            # Score candidates for opening a new route
            # ------------------------------------------------------------------
            # Use K-NN of the depot as candidates: nodes closest to the depot
            # are cheapest to anchor as new-route starters.
            depot_candidates = (set(knn.get(0, [])) & node_set) - visited

            for node in depot_candidates:
                node_waste = wastes.get(node, 0.0)
                if node_waste > cap:
                    continue  # Node alone exceeds capacity

                delta_new = float(dm[0][node] + dm[node][0])
                node_revenue = node_waste * self.R

                # Use compiled lambda for speed
                score = func(
                    node_profit=node_revenue,
                    distance_to_route=dm[0][node],
                    insertion_cost=delta_new,
                    remaining_capacity=cap - node_waste,
                )

                if score > best_score:
                    best_score = score
                    best_action = (node, len(routes), 0)

            # ------------------------------------------------------------------
            # Execute best insertion or handle mandatory fallback / stop
            # ------------------------------------------------------------------
            if best_action is None:
                # Force-insert any mandatory nodes not yet visited
                unvisited_mandatory = (mandatory & node_set) - visited
                if not unvisited_mandatory:
                    break

                # Forced insertion: pick mandatory node with smallest depot cost
                forced_node = min(
                    unvisited_mandatory,
                    key=lambda n: float(dm[0][n] + dm[n][0]),
                )
                routes.append([forced_node])
                route_loads.append(wastes.get(forced_node, 0.0))
                visited.add(forced_node)
                continue

            ins_node, ins_ri, ins_pos = best_action
            if ins_ri == len(routes):
                routes.append([ins_node])
                route_loads.append(wastes.get(ins_node, 0.0))
            else:
                routes[ins_ri].insert(ins_pos, ins_node)
                route_loads[ins_ri] += wastes.get(ins_node, 0.0)

            visited.add(ins_node)

        return [r for r in routes if r]

    def _build_insertion_context(
        self,
        node_revenue: float,
        route: List[int],
        node: int,
        insertion_cost: float,
        dm: np.ndarray,
        remaining_capacity: float,
    ) -> Dict[str, float]:
        """
        Build the local context dictionary for scoring a candidate insertion.

        All four keys are synchronised with ``tree._TERMINALS``.

        Args:
            node_revenue: Revenue of the candidate node (wastes[n] × R).
            route: Current route node list (may be empty for new routes).
            node: Candidate node index.
            insertion_cost: Delta route distance at cheapest position.
            dm: Distance matrix.
            remaining_capacity: Capacity left after insertion.

        Returns:
            Context dictionary with 4 terminal features.
        """
        return {
            "node_profit": node_revenue,
            "distance_to_route": self._min_distance_to_route(node, route, dm),
            "insertion_cost": insertion_cost,
            "remaining_capacity": remaining_capacity,
        }

    # ------------------------------------------------------------------
    # Cheapest insertion helper
    # ------------------------------------------------------------------

    def _cheapest_insertion(self, route: List[int], node: int, dm: np.ndarray) -> Tuple[int, float]:
        """
        Find the position in a route that minimises insertion cost delta.

        Evaluates every inter-node gap (including before-first and after-last)
        and returns the position with the smallest distance increase.

        For an empty route, cost = depot→node + node→depot.

        Args:
            route: Current route node list.
            node: Candidate node to insert.
            dm: Distance matrix.

        Returns:
            Tuple of (best_position, delta_cost).
        """
        if not route:
            return 0, float(dm[0][node] + dm[node][0])

        best_pos = 0
        best_delta = float(dm[0][node] + dm[node][route[0]] - dm[0][route[0]])

        for i in range(len(route) - 1):
            delta = float(dm[route[i]][node] + dm[node][route[i + 1]] - dm[route[i]][route[i + 1]])
            if delta < best_delta:
                best_delta = delta
                best_pos = i + 1

        delta = float(dm[route[-1]][node] + dm[node][0] - dm[route[-1]][0])
        if delta < best_delta:
            best_delta = delta
            best_pos = len(route)

        return best_pos, best_delta

    @staticmethod
    def _min_distance_to_route(node: int, route: List[int], dm: np.ndarray) -> float:
        """Min distance from a candidate node to any node already in the route."""
        if not route:
            return float(dm[0][node])
        return float(min(dm[node][n] for n in route))

    # ------------------------------------------------------------------
    # Fitness evaluation (true generalization)
    # ------------------------------------------------------------------

    def _resolve_training(self) -> List[TrainingEnv]:
        """
        Return training environments: external list or node-subset fallback.

        If ``training_environments`` was supplied to the constructor, use those
        directly — they provide distinct spatial topologies for generalization.

        Otherwise, fall back to sampling node subsets from the test instance
        (same distance matrix, different node configurations).

        Returns:
            List of (dist_matrix, wastes, mandatory_nodes) triples.
        """
        if self.training_environments:
            return self.training_environments

        # Fallback: node-subset sampling on the test distance matrix
        k = max(self.params.n_training_instances, 1)
        ratio = max(0.1, min(1.0, self.params.training_sample_ratio))
        sample_size = max(1, int(len(self.nodes) * ratio))
        instances: List[TrainingEnv] = []

        for _ in range(k):
            optional = [n for n in self.nodes if n not in self.mandatory_nodes]
            n_optional = max(0, sample_size - len(self.mandatory_nodes))
            sampled_opt = self.rng.sample(optional, min(n_optional, len(optional)))
            sampled = list(self.mandatory_nodes) + sampled_opt
            sub_wastes = {n: self.wastes.get(n, 0.0) for n in sampled}
            instances.append((self.dist_matrix, sub_wastes, list(self.mandatory_nodes)))

        return instances

    def _evaluate_tree(
        self,
        tree: GPNode,
        training: List[TrainingEnv],
    ) -> float:
        """
        Evaluate a GP tree's fitness across training environments.

        For each training environment ``(dm, wastes, mandatory)``:
          1. Build a K-NN index for that environment's distance matrix.
          2. Construct a solution using the GP tree's scoring.
          3. Compute normalised profit = profit / max_possible_revenue.

        Fitness is the mean normalised profit minus parsimony penalty:

            fitness = (1/K) Σ_k (profit_k / max_revenue_k) − λ·|tree|

        Args:
            tree: GP scoring function to evaluate.
            training: List of (dist_matrix, wastes, mandatory_nodes) triples.

        Returns:
            Average normalised fitness (float).
        """
        total_normalised = 0.0

        for dm, wastes, mandatory_list in training:
            env_nodes = list(wastes.keys())
            mandatory_set = set(mandatory_list)

            # Build K-NN for this training environment's spatial topology
            env_knn = self._build_knn(dm, env_nodes, self.params.candidate_list_size)

            routes = self._construct_solution(tree, env_nodes, wastes, mandatory_set, dm, env_knn)
            profit = self._evaluate_routes(routes, wastes, dm)
            max_revenue = sum(w * self.R for w in wastes.values())
            total_normalised += profit / max(max_revenue, 1e-9)

        avg = total_normalised / max(len(training), 1)
        avg -= self.params.parsimony_coefficient * tree.size()
        return avg

    # ------------------------------------------------------------------
    # Tournament selection
    # ------------------------------------------------------------------

    def _tournament(self, pop: List[GPNode], fitness: List[float]) -> GPNode:
        """Tournament selection from the GP population."""
        k = min(self.params.tournament_size, len(pop))
        candidates = self.rng.sample(range(len(pop)), k)
        best = max(candidates, key=lambda i: fitness[i])
        return pop[best]

    # ------------------------------------------------------------------
    # Solution evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate_routes(self, routes: List[List[int]], wastes: Dict[int, float], dm: np.ndarray) -> float:
        """Net profit: Σ(wastes[n] × R) − distance × C."""
        if not routes:
            return 0.0
        rev = sum(wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes, dm) * self.C

    def _cost(self, routes: List[List[int]], dm: np.ndarray) -> float:
        """Total routing distance across all routes."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += dm[0][route[0]]
            for i in range(len(route) - 1):
                total += dm[route[i]][route[i + 1]]
            total += dm[route[-1]][0]
        return float(total)
