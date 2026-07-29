import itertools
from typing import Dict, List, Tuple, Union

import numpy as np


class StateSpaceManager:
    """
    Manages the discrete state space and transition probabilities for the Exact SDP.

    The state is a tuple of V integers, each in {0, 1, ..., L-1}.
    """

    def __init__(self, num_nodes: int, discrete_levels: int, max_fill_rate: float):
        self.num_nodes = num_nodes
        self.L = discrete_levels
        # Build 1D transition probabilities for a single bin
        self._build_bin_transitions(max_fill_rate)

    def get_all_states(self) -> List[Tuple[int, ...]]:
        """Return a list of all possible discrete state tuples."""
        # Excluding depot, depot has no fill level
        customer_nodes = self.num_nodes - 1
        return list(itertools.product(range(self.L), repeat=customer_nodes))

    def _build_bin_transitions(self, mean_increment: float):
        """
        Builds a discrete probability transition matrix for a single bin.
        P[l][l'] = Probability of moving from level l to l' without a visit.
        If a visit occurs, the bin resets to l=0 -> we just use P[0][l'].

        We construct a simple discretized, discretized-gamma or uniformly spread distribution
        where the expected increment matches mean_increment.
        mean_increment is roughly continuous [0, 1]. In discrete units, it is expected_shift = mean_increment * (L-1).
        """
        expected_shift = mean_increment * (self.L - 1)
        # simplistic Poisson-like logic or simply distributing mass
        # For an exact benchmark, the exact shape isn't as critical as having *a* known transition matrix.
        self.trans_matrix = np.zeros((self.L, self.L))

        # Base discrete distribution for delta
        dist = np.zeros(self.L * 2)
        # For simplicity, assign probability to floor(expected_shift) and ceil(expected_shift)
        f_shift = int(np.floor(expected_shift))
        c_shift = int(np.ceil(expected_shift))
        if f_shift == c_shift:
            dist[f_shift] = 1.0
        else:
            p_c = expected_shift - f_shift
            p_f = 1.0 - p_c
            dist[f_shift] = p_f
            dist[c_shift] = p_c

        # Normalize Dist (just in case bounds are wide)
        dist = dist / dist.sum()

        for l in range(self.L):
            for i, p in enumerate(dist):
                if p > 0:
                    l_prime = min(self.L - 1, l + i)
                    self.trans_matrix[l, l_prime] += p

    def get_transition_probs(
        self, state: Tuple[int, ...], action_set: Union[frozenset, set]
    ) -> List[Tuple[Tuple[int, ...], float]]:
        """
        Given a state and an action (subset of nodes to visit),
        generate reachable non-zero-prob states and their probabilities.

        Note: action_set contains global node indices. Customers start at 1.
        So bin index in state tuple is (node_idx - 1).
        """
        bin_probs = []
        for b_idx in range(self.num_nodes - 1):
            node_idx = b_idx + 1
            curr_l = 0 if node_idx in action_set else state[b_idx]  # Re-set bin to empty if collected

            # Possible next levels and their probabilities
            probs = [
                (nxt_l, self.trans_matrix[curr_l, nxt_l])
                for nxt_l in range(self.L)
                if self.trans_matrix[curr_l, nxt_l] > 0
            ]
            bin_probs.append(probs)

        reachable_states = []
        for combo in itertools.product(*bin_probs):
            st = tuple(val for val, _ in combo)
            p = float(np.prod([val for _, val in combo]))
            if p > 0:
                reachable_states.append((st, p))

        return reachable_states

    def state_to_fraction(self, state: Tuple[int, ...]) -> Dict[int, float]:
        """Map discrete state tuple back to dictionary of continuous variables."""
        return {b_idx + 1: state[b_idx] / max(1, self.L - 1) for b_idx in range(self.num_nodes - 1)}

    def fraction_to_state(self, fractions: Dict[int, float]) -> Tuple[int, ...]:
        """Map continuous dictionary to discrete state tuple."""
        lst = []
        for b_idx in range(self.num_nodes - 1):
            node_idx = b_idx + 1
            val = fractions.get(node_idx, 0.0)
            discrete_val = int(round(val * (self.L - 1)))
            discrete_val = max(0, min(self.L - 1, discrete_val))
            lst.append(discrete_val)
        return tuple(lst)
