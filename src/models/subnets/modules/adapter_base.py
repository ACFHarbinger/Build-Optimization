"""Base class for GLOP subproblem adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import torch
from tensordict import TensorDict

from .subproblem_mapping import SubproblemMapping


class SubproblemAdapter(ABC):
    """
    Base class for adapters that convert global partitions to local subproblems.
    """

    subproblem_env_name: str = "tsp"

    def __init__(
        self,
        td: TensorDict,
        partition_actions: torch.Tensor,
        subprob_batch_size: int = 2000,
    ) -> None:
        """
        Initialize adapter with problem instance and partition.
        """
        self.td = td
        self.partition_actions = partition_actions
        self.subprob_batch_size = subprob_batch_size

        # Store original coordinates
        self.coordinates = td.get("locs", td.get("locations"))
        self.batch_size = self.coordinates.size(0)

        # Initialize storage for final actions
        self.final_actions = torch.zeros_like(partition_actions)

    @abstractmethod
    def get_batched_subprobs(self) -> Iterator[SubproblemMapping]:
        """
        Yield batches of subproblems for solving.
        """
        pass

    @abstractmethod
    def update_actions(
        self,
        mapping: SubproblemMapping,
        subprob_actions: torch.Tensor,
    ) -> None:
        """
        Update final actions with solved subproblem.
        """
        pass

    def get_actions(self) -> torch.Tensor:
        """Get final actions after all subproblems solved."""
        return self.final_actions
