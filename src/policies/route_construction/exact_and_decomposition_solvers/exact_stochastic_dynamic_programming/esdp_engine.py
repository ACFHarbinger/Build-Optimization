from typing import Dict, FrozenSet, Tuple

import numpy as np
from tqdm import tqdm

from .cop_evaluator import COPEvaluator
from .params import SDPParams
from .state_space import StateSpaceManager


class ExactSDPEngine:
    """
    Executes the backward induction for the exact SDP.
    """

    def __init__(self, params: SDPParams, dist_matrix: np.ndarray, capacity: float):
        self.params = params
        self.num_nodes = dist_matrix.shape[0]
        self.capacity = capacity

        self.state_space = StateSpaceManager(self.num_nodes, params.discrete_levels, params.max_fill_rate)
        self.cop_eval = COPEvaluator(dist_matrix, self.num_nodes)

        # Policy Table: Day -> State -> Optimal Action (Subset of nodes)
        self.policy_table: Dict[int, Dict[Tuple[int, ...], FrozenSet[int]]] = {}
        # Value Table: Day -> State -> Expected Value
        self.value_table: Dict[int, Dict[Tuple[int, ...], float]] = {}

    def _compute_overflow_penalty(self, state: Tuple[int, ...]) -> float:
        """Calculate overflow penalties for the current state."""
        overflows = sum(1 for val in state if val == self.params.discrete_levels - 1)
        return overflows * self.params.overflow_penalty

    def _evaluate_action(
        self, state: Tuple[int, ...], action: FrozenSet[int], next_val_table: Dict[Tuple[int, ...], float]
    ) -> float:
        """Evaluate the Q-value of a state-action pair."""
        # 1. Routing Cost
        routing_cost = self.cop_eval.get_route_cost(action)

        # 2. Collected Waste
        # Assuming discrete levels map linearly to [0, 1] capacity space
        collected = sum((state[i - 1] / max(1, self.params.discrete_levels - 1)) for i in action)

        # 3. Overflow Penalties
        overflow_cost = self.compute_overflow_penalty(state)

        stage_reward = (self.params.waste_weight * collected) - (self.params.cost_weight * routing_cost) - overflow_cost

        # 4. Expected Future Value
        expected_future = 0.0
        if next_val_table:
            reachable_states = self.state_space.get_transition_probs(state, action)
            for nxt_st, prob in reachable_states:
                expected_future += prob * next_val_table.get(nxt_st, 0.0)

        return stage_reward + self.params.discount_factor * expected_future

    def compute_overflow_penalty(self, state: Tuple[int, ...]) -> float:
        # duplicated safe call for overflow
        return self._compute_overflow_penalty(state)

    def solve(self):
        """Perform exact backward induction from Day D down to Day 1."""
        all_states = self.state_space.get_all_states()

        # Initialize terminal condition V_{D+1} = 0
        next_val_table = {s: 0.0 for s in all_states}

        for day in range(self.params.num_days, 0, -1):
            curr_val_table = {}
            curr_policy_table = {}

            # For progress visualization (optional, can be disabled if too fast or small)
            print(f"Solving SDP for Day {day}/{self.params.num_days} across {len(all_states)} states...")

            for state in tqdm(all_states, leave=False):
                best_val = -float("inf")
                best_action = frozenset()

                valid_actions = self.cop_eval.get_feasible_actions(state, self.capacity, self.params.discrete_levels)

                for action in valid_actions:
                    q_val = self._evaluate_action(state, action, next_val_table)
                    if q_val > best_val:
                        best_val = q_val
                        best_action = action

                curr_val_table[state] = best_val
                curr_policy_table[state] = best_action

            self.value_table[day] = curr_val_table
            self.policy_table[day] = curr_policy_table
            next_val_table = curr_val_table

        print("SDP Backward Induction complete.")

    def get_optimal_action(self, day: int, state: Tuple[int, ...]) -> FrozenSet[int]:
        """Return the optimal subset of nodes to visit."""
        day = max(1, min(self.params.num_days, day))
        # Snap state to nearest discrete bounds just in case
        safe_state = tuple(max(0, min(self.params.discrete_levels - 1, int(v))) for v in state)
        return self.policy_table[day].get(safe_state, frozenset())
