"""TSP Adapter for GLOP."""

from __future__ import annotations

from typing import Iterator, List

import torch
from tensordict import TensorDict

from .adapter_base import SubproblemAdapter
from .subproblem_mapping import SubproblemMapping


class TSPAdapter(SubproblemAdapter):
    """
    TSP Adapter for converting partitions to TSP subproblems.
    """

    subproblem_env_name: str = "tsp"

    def __init__(
        self,
        td: TensorDict,
        partition_actions: torch.Tensor,
        subprob_batch_size: int = 2000,
        **kwargs,
    ) -> None:
        """
        Initialize TSPAdapter.
        """
        super().__init__(td, partition_actions, subprob_batch_size)

        # Extract subproblems from partitions
        self._subproblems: List[SubproblemMapping] = []
        self._extract_subproblems()

    def _extract_subproblems(self) -> None:
        """Extract TSP subproblems from partition actions."""
        for batch_idx in range(self.batch_size):
            coords = self.coordinates[batch_idx]
            actions = self.partition_actions[batch_idx]

            # Group nodes by partition
            unique_partitions = actions.unique()

            for route_idx, partition_id in enumerate(unique_partitions):
                # Get nodes in this partition
                mask = actions == partition_id
                indices = torch.where(mask)[0]

                if len(indices) < 2:
                    continue

                # Get coordinates for this partition
                subprob_coords = coords[indices]

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
        batch = []
        for subprob in self._subproblems:
            batch.append(subprob)
            if len(batch) >= self.subprob_batch_size:
                for item in batch:
                    yield item
                batch = []

        # Yield remaining
        for item in batch:
            yield item

    def update_actions(
        self,
        mapping: SubproblemMapping,
        subprob_actions: torch.Tensor,
    ) -> None:
        """Update final actions with TSP solution."""
        # Map subproblem solution back to original indices
        original_indices = mapping.original_indices
        batch_idx = mapping.batch_idx

        # The subprob_actions gives tour order within the subproblem
        # We need to translate this to the original action format
        for pos, node_idx in enumerate(subprob_actions):
            if node_idx < len(original_indices):
                original_node = original_indices[node_idx]
                self.final_actions[batch_idx, pos] = original_node
