from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ScenarioNode:
    """A node in the scenario tree."""

    id: int
    day: int
    probability: float
    # Realization of demand increments for this node relative to its parent
    realization: Dict[int, float]
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)


class ScenarioTree:
    """
    Constructs and manages a branching scenario tree for multi-day stochastic optimization.
    """

    def __init__(
        self,
        num_days: int,
        num_realizations: int,
        customers: List[int],
        mean_increment: float,
        seed: Optional[int] = 42,
    ):
        self.num_days = num_days
        self.num_realizations = num_realizations
        self.customers = customers
        self.mean_increment = mean_increment
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.nodes: Dict[int, ScenarioNode] = {}
        self._build_tree()

    def _build_tree(self):
        """Build the tree structure."""
        # 1. Generate 'num_realizations' joint scenarios for a SINGLE day
        # These will be reused at each branching point.
        daily_scenarios = []
        for _ in range(self.num_realizations):
            # Sample independent Gamma-like increments for each customer
            # For simplicity, we use Gamma(alpha=2, beta=mean/2) for some variance
            # or just simple normal/uniform.
            scen = {}
            for node_idx in self.customers:
                # α=4 -> CV=0.5
                alpha = 4.0
                beta = self.mean_increment / alpha
                val = self.rng.gamma(alpha, beta)
                scen[node_idx] = float(np.clip(val, 0.0, 1.0))
            daily_scenarios.append(scen)

        # 2. Iteratively expand the tree
        # Root node (Day 0 - Initial State revealed)
        root = ScenarioNode(id=0, day=0, probability=1.0, realization={i: 0.0 for i in self.customers})
        self.nodes[0] = root

        current_layer = [0]
        node_counter = 1

        for d in range(1, self.num_days + 1):
            next_layer = []
            for parent_id in current_layer:
                parent_prob = self.nodes[parent_id].probability

                for s_idx in range(self.num_realizations):
                    child_id = node_counter
                    node_counter += 1

                    node = ScenarioNode(
                        id=child_id,
                        day=d,
                        probability=parent_prob / self.num_realizations,
                        realization=daily_scenarios[s_idx],
                        parent_id=parent_id,
                    )
                    self.nodes[child_id] = node
                    self.nodes[parent_id].children_ids.append(child_id)
                    next_layer.append(child_id)

            current_layer = next_layer

    def get_nodes_by_day(self, day: int) -> List[ScenarioNode]:
        """Return all nodes at a specific day."""
        return [n for n in self.nodes.values() if n.day == day]

    def get_root(self) -> ScenarioNode:
        return self.nodes[0]

    def get_leaves(self) -> List[ScenarioNode]:
        return [n for n in self.nodes.values() if not n.children_ids]
