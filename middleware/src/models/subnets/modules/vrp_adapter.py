"""VRP Adapter for GLOP."""

from __future__ import annotations

from typing import Iterator, List, Optional

import torch
from tensordict import TensorDict

from .adapter_base import SubproblemAdapter
from .subproblem_mapping import SubproblemMapping


class VRPAdapter(SubproblemAdapter):
    """
    VRP Adapter for converting partitions to VRP-style subproblems.
    """

    subproblem_env_name: str = "cvrp"

    def __init__(
        self,
        td: TensorDict,
        partition_actions: torch.Tensor,
        subprob_batch_size: int = 2000,
        capacity: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Initialize VRPAdapter.
        """
        super().__init__(td, partition_actions, subprob_batch_size)

        self.capacity = capacity or td.get("vehicle_capacity", torch.tensor(1.0)).item()
        self._subproblems: List[SubproblemMapping] = []
        self._extract_subproblems()

    def _extract_subproblems(self) -> None:
        """Extract VRP subproblems from partition actions."""
        for batch_idx in range(self.batch_size):
            coords = self.coordinates[batch_idx]
            actions = self.partition_actions[batch_idx]

            # First node is depot
            depot = coords[0:1]

            # Group nodes by route assignment
            unique_routes = actions.unique()

            for route_idx, route_id in enumerate(unique_routes):
                mask = actions == route_id
                indices = torch.where(mask)[0]

                # Skip depot-only routes
                if len(indices) < 1:
                    continue

                # Include depot at start
                subprob_coords = torch.cat([depot, coords[indices]], dim=0)

                self._subproblems.append(
                    SubproblemMapping(
                        batch_idx=batch_idx,
                        sample_idx=0,
                        subprob_coordinates=subprob_coords,
                        original_indices=indices,
                        route_idx=route_idx,
                    )
                )

    def get_batched_subprobs(self) -> Iterator[SubproblemMapping]:
        """Yield subproblems in batches."""
        for subprob in self._subproblems:
            yield subprob

    def update_actions(
        self,
        mapping: SubproblemMapping,
        subprob_actions: torch.Tensor,
    ) -> None:
        """Update final actions with VRP route solution."""
        original_indices = mapping.original_indices
        batch_idx = mapping.batch_idx

        # Map subproblem actions (which include depot) back to original
        for pos, node_idx in enumerate(subprob_actions):
            if node_idx == 0:
                # Depot - use 0 in final actions
                self.final_actions[batch_idx, pos] = 0
            elif node_idx <= len(original_indices):
                original_node = original_indices[node_idx - 1]
                self.final_actions[batch_idx, pos] = original_node
