"""
Hidden Markov Model + Great Deluge Hyper-Heuristic (HMM-GD-HH) for VRPP.

This online-learning approach treats the sequence of applied Low-Level
Heuristics (LLHs) as a Markov chain. The system maintains a *belief*
distribution over three hidden states — improving, stagnating, escaping —
using the Forward Algorithm, rather than assigning a single deterministic
state.

At each iteration the belief vector P(S_t | O_{1:t}) is updated via the
observation likelihood of the normalised profit change Δ_norm.  The LLH
selection probabilities are then the belief-weighted mixture of the
per-state emission matrices.

The Great Deluge acceptance criterion accepts candidate solutions whose
profit exceeds a linearly rising water level, providing a deterministic
escape from local optima without requiring temperature tuning.

Reference:
    Onsem, W. V., Demoen, B., & Causmaecker, P. D. "HHaaHMM:
    A Hyper-Heuristic as a Hidden Markov Model", 2014
    McMullan, P. "An Extended Implementation of the Great
    Deluge Algorithm for Course Timetabling", 2007
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators import (
    cluster_removal,
    greedy_insertion,
    greedy_profit_insertion,
    random_removal,
    regret_2_insertion,
    regret_2_profit_insertion,
    shaw_profit_removal,
    shaw_removal,
    string_removal,
    worst_profit_removal,
    worst_removal,
)
from logic.src.policies.helpers.operators.solution_initialization.nearest_neighbor_si import build_nn_routes

from .params import HMMGDHHParams

# Discrete Observation Alphabet (Onsem et al. 2014, Section 5.1)
_OBS_IMPROVE = 0  # ΔProfit > 0
_OBS_WORSE = 1  # ΔProfit < 0, ΔCost = 0
_OBS_WORSE_COST = 2  # Fix #R-Final-1 (Semantic Renaming)
_OBS_SAME = 3  # ΔProfit = 0, ΔCost = 0
_OBS_SAME_COST = 4  # Fix #R-Final-1 (Semantic Renaming)
_N_OBS = 5


class HMMGDHHSolver:
    r"""
    Hidden Markov Model + Great Deluge Hyper-Heuristic solver (HHaaHMM).

    Aligned with: Onsem, W. V., Demoen, B., & Causmaecker, P. D. "HHaaHMM:
    A Hyper-Heuristic as a Hidden Markov Model", 2014.

    This implementation treats the hidden states as representing the **local topology
    of the search space** evaluated by the heuristics. This topology is modeled as
    an Input-Output Hidden Markov Model (IOHMM), where:
    - **Inputs ($u_t$)**: Low-Level Heuristics (LLHs).
    - **Outputs ($o_t$)**: Discrete categories of solution change ($\Delta$ Profit, $\Delta$ Cost).
    - **Hidden States ($s_t$)**: Structural search phases (e.g., local optima regions).

    Bi-Objective Transformation (Fix #R-Final-1):
    This implementation adapts the original IOHMM observation alphabet. Instead of
    tracking computational time, we repurpose the 'time' dimension to track
    'routing cost' (distance). This transforms the HMM into a bi-objective model
    that learns the trade-off between financial profit and physical routing efficiency.

    Note on Great Deluge Integration:
    The IOHMM learns and updates its belief based on *every* proposed move (exploration),
    mapping the immediate neighborhood dynamics. The Great Deluge (GD) criterion
    independently controls the *acceptance trajectory*.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: HMMGDHHParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

        # Dynamic LLH Pool (Fix #R-Final-2: Ablation Blocker)
        # Sliced according to params.n_llh to enable proper research evaluation
        full_pool = [self._llh0, self._llh1, self._llh2, self._llh3, self._llh4, self._llh5, self._llh6]
        self._llh_pool = full_pool[: params.n_llh]
        self.n_llh = len(self._llh_pool)

        # Static parameters
        self.alpha = 0.5  # Weight for entropy in action selection
        self.gamma = 0.9  # Decay factor for online EM counts (optional smoothing)

        # --- Dynamic HHaaHMM State ---

        # Number of hidden states starts small (Fix #3)
        self.n_states = 2
        self.t = 0  # Observation counter

        # Transition matrix A[s_t][s_{t+1}] | input u_t (HHaaHMM uses P(s' | s, u))
        # Initialised with small random noise to break symmetry (Fix #R-Final-2)
        self._A = np.ones((self.n_llh, self.n_states, self.n_states)) / self.n_states
        self._A += np.array(
            [
                [[self.random.uniform(-0.01, 0.01) for _ in range(self.n_states)] for _ in range(self.n_states)]
                for _ in range(self.n_llh)
            ]
        )
        self._A = np.clip(self._A, 1e-12, 1.0)
        self._A /= self._A.sum(axis=2, keepdims=True)

        # Emission matrix B[s_t][o_t] | input u_t (HHaaHMM uses P(o | s, u))
        # Initialised with small random noise to break symmetry (Fix #R-Final-2)
        self._B = np.ones((self.n_llh, self.n_states, _N_OBS)) / _N_OBS
        self._B += np.array(
            [
                [[self.random.uniform(-0.01, 0.01) for _ in range(_N_OBS)] for _ in range(self.n_states)]
                for _ in range(self.n_llh)
            ]
        )
        self._B = np.clip(self._B, 1e-12, 1.0)
        self._B /= self._B.sum(axis=2, keepdims=True)

        # State belief: P(state | history)
        self._belief = np.ones(self.n_states) / self.n_states

        # EM counters (for stochastic online EM estimation)
        self._counts_A = np.zeros((self.n_llh, self.n_states, self.n_states))
        self._counts_B = np.zeros((self.n_llh, self.n_states, _N_OBS))

        # --- Refinement 1: State-Aware Expected Profit (Fix #1) ---
        # Matrix shape: (n_states, n_llh)
        self._expected_profit = np.zeros((self.n_states, self.n_llh))
        self._profit_counts = np.zeros((self.n_states, self.n_llh))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, initial_routes: List[List[int]]) -> Tuple[List[List[int]], float, float]:
        """
        Main HH-GD loop for VRPP.

        Following the principle of "calculating a good heuristic at any point in time
        given the old data" (Onsem et al. 2014, Section 6), the IOHMM learns the
        local neighborhood topology from every proposed move. The Great Deluge
        threshold then filters these proposals to manage the global search trajectory.
        """
        routes = copy.deepcopy(initial_routes)
        profit = self._evaluate(routes)
        best_routes = copy.deepcopy(routes)
        best_profit = profit
        best_cost = self._cost(routes)

        start_time = time.process_time()

        # Great Deluge: water level starts linearly rising
        water_level = (
            best_profit * (1.0 - self.params.flood_margin) if best_profit > 0 else -abs(self.params.flood_margin)
        )

        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # 1. Action Selection: Entropy-Maximizing (Fix #5, #R1, #R-Final-1 & #Final-3)
            # score(u) = NormalizedProfit(u) + annealed_alpha * Entropy(P(s'|u))
            u_idx = self._select_action(iteration, self.params.max_iterations)
            llh = self._llh_pool[u_idx]

            # 2. Dynamic State Scaling (Fix #3)
            self.t += 1
            self._check_state_scaling()

            # Store belief prior to action (for state-aware profit update)
            previous_belief = self._belief.copy()

            # 3. Apply LLH and observe outcomes
            # The HMM learns from all moves (exploration), uncoupled from Great Deluge acceptance (Fix #R3)
            old_cost = self._cost(routes)
            try:
                new_routes = llh(routes, self.params.n_removal)
                new_profit = self._evaluate(new_routes)
                new_cost = self._cost(new_routes)
            except Exception:
                new_routes = routes
                new_profit = profit
                new_cost = old_cost

            # 4. Map to discrete observation alphabet (Fix #R-Final-1)
            o_idx = self._map_observation(new_profit - profit, new_cost - old_cost)

            # 5. Online EM update (Fix #R-Final-3: Stochastic Online EM)
            self._online_em_update(u_idx, o_idx)

            # 6. Refinement 1: State-Aware Profit Update (Fix #R1 & #R-Final-2)
            # update E[p | s, u] for all s weighted by P(s | history)
            delta_profit = new_profit - profit

            # --- Fix #R-Final-2: Wasted Effort Penalty ---
            # If the move is idempotent (SAME or SAME_COST), apply a small penalty
            # even if delta_profit is 0, to encourage search progression.
            if o_idx in [_OBS_SAME, _OBS_SAME_COST]:
                delta_profit -= 1e-3

            for s_idx in range(self.n_states):
                self._profit_counts[s_idx, u_idx] += previous_belief[s_idx]
                # Incremental online average of profit weighted by state belief
                step = previous_belief[s_idx] / (self._profit_counts[s_idx, u_idx] + 1e-12)
                self._expected_profit[s_idx, u_idx] += step * (delta_profit - self._expected_profit[s_idx, u_idx])

            # Great Deluge acceptance
            if new_profit >= water_level:
                routes = new_routes
                profit = new_profit
                if profit > best_profit:
                    best_routes = copy.deepcopy(routes)
                    best_profit = profit
                    best_cost = self._cost(best_routes)

            # Increase water level
            water_level += self.params.rain_speed * abs(best_profit + 1e-9)

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                water_level=water_level,
                hmm_state=int(np.argmax(self._belief)),
                llh_selected=u_idx,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # HHaaHMM Logic
    # ------------------------------------------------------------------

    def _map_observation(self, delta_profit: float, delta_cost: float) -> int:
        """Map profit/cost changes to discrete alphabet (Fix #R-Final-1)."""
        if delta_profit > 1e-9:
            return _OBS_IMPROVE
        if delta_profit < -1e-9:
            return _OBS_WORSE_COST if delta_cost > 1e-9 else _OBS_WORSE
        # delta_profit is roughly 0
        return _OBS_SAME_COST if delta_cost > 1e-9 else _OBS_SAME

    def _check_state_scaling(self):
        """Scale number of states with O(sqrt(log t)) (Fix #3)."""
        target_n = int(np.ceil(np.sqrt(np.log(self.t + 1)))) + 1
        if target_n > self.n_states:
            self._split_state()

    def _split_state(self):
        """
        Split the "least determined" state (Fix #1: Combined Entropy Split).

        Following Onsem et al. (2014) Section 6, the state with the highest combined
        uncertainty in both output character and next state is chosen for splitting.
        """
        uncertainties = []
        for s in range(self.n_states):
            # Emission entropy H(B|s) summed over LLHs
            h_b = -np.sum(self._B[:, s, :] * np.log(self._B[:, s, :] + 1e-12))
            # Transition entropy H(A|s) summed over LLHs
            h_a = -np.sum(self._A[:, s, :] * np.log(self._A[:, s, :] + 1e-12))
            uncertainties.append(h_b + h_a)

        s_split = int(np.argmax(uncertainties))

        # Increment n_states
        old_n = self.n_states
        new_n = old_n + 1
        self.n_states = new_n

        # Resize Matrices (A is [u, s, s'], B is [u, s, o])
        new_A = np.ones((self.n_llh, new_n, new_n)) / new_n
        new_B = np.ones((self.n_llh, new_n, _N_OBS)) / _N_OBS

        # Copy old values
        new_A[:, :old_n, :old_n] = self._A
        new_B[:, :old_n, :] = self._B

        # --- Refinement 2: Transition Matrix Inheritance (Fix #R2) ---
        # The new state inherits outgoing transitions from s_split
        new_A[:, new_n - 1, :old_n] = self._A[:, s_split, :]
        # The new state inherits incoming transitions to s_split
        new_A[:, :old_n, new_n - 1] = self._A[:, :, s_split]
        # Self-transition for the new state inherits from parent
        new_A[:, new_n - 1, new_n - 1] = self._A[:, s_split, s_split]

        # Apply small random perturbations to break symmetry (Transitions)
        perturb_a = np.array(
            [
                [[self.random.uniform(-0.01, 0.01) for _ in range(new_n)] for _ in range(new_n)]
                for _ in range(self.n_llh)
            ]
        )
        new_A[:, new_n - 1, :] += perturb_a[:, new_n - 1, :]
        new_A = np.clip(new_A, 1e-12, 1.0)
        new_A /= new_A.sum(axis=2, keepdims=True)

        # Apply small random perturbations (Emissions)
        perturb_b = np.array([[self.random.uniform(-0.01, 0.01) for _ in range(_N_OBS)] for _ in range(self.n_llh)])
        new_B[:, new_n - 1, :] = self._B[:, s_split, :] + perturb_b
        new_B = np.clip(new_B, 1e-12, 1.0)
        new_B /= new_B.sum(axis=2, keepdims=True)

        self._A = new_A
        self._B = new_B
        self._counts_A = np.zeros((self.n_llh, new_n, new_n))
        self._counts_B = np.zeros((self.n_llh, new_n, _N_OBS))

        # --- Resize State-Aware Profit Matrices (Fix #R1) ---
        new_expected_profit = np.zeros((new_n, self.n_llh))
        new_profit_counts = np.zeros((new_n, self.n_llh))
        new_expected_profit[:old_n, :] = self._expected_profit
        new_profit_counts[:old_n, :] = self._profit_counts
        # New state inherits profit expectations from s_split
        new_expected_profit[new_n - 1, :] = self._expected_profit[s_split, :]
        # Split history counts
        new_profit_counts[new_n - 1, :] = self._profit_counts[s_split, :] / 2.0
        self._profit_counts[s_split, :] /= 2.0
        self._expected_profit = new_expected_profit
        self._profit_counts = new_profit_counts

        # Resize belief
        new_belief = np.zeros(new_n)
        new_belief[:old_n] = self._belief * 0.5
        new_belief[s_split] = self._belief[s_split] * 0.5
        new_belief[new_n - 1] = self._belief[s_split]  # shared belief
        self._belief = new_belief / new_belief.sum()

    def _online_em_update(self, u_idx: int, o_idx: int):
        """
        Stochastic Online EM approximation for HHaaHMM (Fix #R-Final-3).

        Due to the continuous streaming nature of the hyper-heuristic, this replaces
        the batch offline Baum-Welch algorithm with a Stochastic Online EM
        approximation. It estimates the joint transition probabilities using
        strictly the Forward variables, applying a decay factor γ=0.9 to
        construct an exponential moving average of the state-transition counts.
        """
        # Forward Step: alpha_t(j) = P(o_t | s_j, u_t) * sum_i alpha_{t-1}(i) * P(s_j | s_i, u_t)
        new_belief = np.zeros(self.n_states)
        for j in range(self.n_states):
            transmission = np.dot(self._belief, self._A[u_idx, :, j])
            new_belief[j] = self._B[u_idx, j, o_idx] * transmission

        belief_sum = new_belief.sum()
        if belief_sum > 1e-12:
            new_belief /= belief_sum
        else:
            new_belief = np.ones(self.n_states) / self.n_states

        # Update counts for online EM (simplified online updates)
        # counts_A(i,j) += P(s_t=i, s_{t+1}=j | history)
        # counts_B(i,o) += P(s_t=i | history and obs o)
        for i in range(self.n_states):
            for j in range(self.n_states):
                # Approximation of joint probability for online use
                joint = self._belief[i] * self._A[u_idx, i, j] * self._B[u_idx, j, o_idx]
                self._counts_A[u_idx, i, j] = self.gamma * self._counts_A[u_idx, i, j] + joint

            self._counts_B[u_idx, i, o_idx] = self.gamma * self._counts_B[u_idx, i, o_idx] + self._belief[i]

        # Re-normalise A and B
        for i in range(self.n_states):
            # Normalise A[u, i, :]
            row_sum_a = self._counts_A[u_idx, i, :].sum()
            if row_sum_a > 1e-12:
                self._A[u_idx, i, :] = self._counts_A[u_idx, i, :] / row_sum_a

            # Normalise B[u, i, :]
            row_sum_b = self._counts_B[u_idx, i, :].sum()
            if row_sum_b > 1e-12:
                self._B[u_idx, i, :] = self._counts_B[u_idx, i, :] / row_sum_b

        self._belief = new_belief

    def _select_action(self, iteration: int, max_iterations: int) -> int:
        """
        Entropy-Maximizing Action Selection with Dynamic Annealing (Fix #5, #R1, #R-Final-1 & #Final-3).

        score(u) = NormalizedProfit(u) + current_alpha * Entropy(P(s_{t+1}|u))
        """
        # 1. Calculate raw expected profits and future entropies
        raw_profits = np.zeros(self.n_llh)
        entropies = np.zeros(self.n_llh)
        for u in range(self.n_llh):
            # Predicted next state distribution for action u: P(s' | u) = sum_i P(s_i|hist) * P(s'|s_i,u)
            p_next = np.dot(self._belief, self._A[u, :, :])
            # Entropy: force distribution towards uniform to explore
            entropies[u] = -np.sum(p_next * np.log(p_next + 1e-12))

            # --- Refinement 1: State-Aware Profit Expectation (Fix #R1) ---
            # E[Profit | u] = sum_s P(s | history) * E[Profit | s, u]
            raw_profits[u] = np.dot(self._belief, self._expected_profit[:, u])

        # 2. Robust Min-Max Normalization (Fix #R-Final-1 & #3: Neutralize Ties)
        # Binds profits to [0, 1] so they are comparable to entropy (bounded by ln(n)).
        # If all profits are equal (delta < 1e-9), set to 0.5 to allow entropy to drive exploration.
        p_min, p_max = raw_profits.min(), raw_profits.max()
        if p_max - p_min > 1e-9:
            normalized_profits = (raw_profits - p_min) / (p_max - p_min)
        else:
            normalized_profits = np.full(self.n_llh, 0.5)

        # 3. Dynamic Exploration Annealing (Fix #R-Final-3)
        # alpha decays linearly to zero to transition from exploration to deep exploitation
        current_alpha = self.alpha * (1.0 - (iteration / max_iterations))

        # 4. Combine normalized profit and annealed entropy exploration bonus
        scores = normalized_profits + current_alpha * entropies

        return int(np.argmax(scores))

    # ------------------------------------------------------------------
    # Forward Algorithm helpers (REMOVED: Old Gaussian PDF logic)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # LLH pool
    # ------------------------------------------------------------------

    def _llh0(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L0: random_removal + greedy_insertion."""
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        p, r = random_removal(routes, n, self.random)
        if use_profit:
            return greedy_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        return greedy_insertion(
            p,
            r,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
        )

    def _llh1(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L1: worst_removal + regret_2_insertion."""
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        if use_profit:
            p, r = worst_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R, self.C)
            return regret_2_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        p, r = worst_removal(routes, n, self.dist_matrix)
        return regret_2_insertion(
            p,
            r,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
        )

    def _llh2(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L2: cluster_removal + greedy_insertion."""
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        p, r = cluster_removal(routes, n, self.dist_matrix, self.nodes)
        if use_profit:
            return greedy_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        return greedy_insertion(
            p,
            r,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
        )

    def _llh3(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L3: worst_removal + greedy_insertion."""
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        if use_profit:
            p, r = worst_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R, self.C)
            return greedy_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        p, r = worst_removal(routes, n, self.dist_matrix)
        return greedy_insertion(
            p,
            r,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
        )

    def _llh4(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L4: random_removal + regret_2_insertion."""
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        p, r = random_removal(routes, n, self.random)
        if use_profit:
            return regret_2_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        return regret_2_insertion(
            p,
            r,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
        )

    def _llh5(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L5: shaw_removal + greedy_insertion."""
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        if use_profit:
            p, r = shaw_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R, self.C, rng=self.random)
            return greedy_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        p, r = shaw_removal(routes, n, self.dist_matrix)
        return greedy_insertion(
            p,
            r,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
        )

    def _llh6(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L6: string_removal + regret_2_insertion."""
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        p, r = string_removal(routes, n, self.dist_matrix, rng=self.random)
        if use_profit:
            return regret_2_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        return regret_2_insertion(
            p,
            r,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_random_solution(self) -> List[List[int]]:
        """Order-dependent sequential construction (matches ALNS style).

        Random node ordering causes different capacity cutoffs, creating
        genuinely diverse initial solutions. Uses self.C for the profitability
        check so that economics are consistent with the solver's _evaluate().
        """
        optimized_routes = build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
        )
        return optimized_routes

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Net profit for a set of routes."""
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """Total routing distance."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
