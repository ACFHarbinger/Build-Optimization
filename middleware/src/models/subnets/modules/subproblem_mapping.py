"""GLOP Subproblem Mapping."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SubproblemMapping:
    """Mapping from subproblem to original problem indices."""

    batch_idx: int
    sample_idx: int
    subprob_coordinates: torch.Tensor
    original_indices: torch.Tensor
    route_idx: int
