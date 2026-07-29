"""
TensorDict State Wrapper for legacy decoder compatibility.
"""

from typing import Optional

import torch
from tensordict import TensorDict


class TensorDictStateWrapper:
    """
    Wraps a TensorDict to expose methods expected by legacy model components (Decoders).

    Acts as a bridge between the new TensorDict state management and the
    legacy object-oriented state access patterns (e.g. state.get_mask()).
    """

    def __init__(self, td: TensorDict, problem_name: str = "vrpp", env=None):
        """Initialize TensorDictStateWrapper."""
        self.td = td
        self.problem_name = problem_name
        self.env = env

        # AttentionDecoder loop expects these
        if "ids" in td.keys():
            self.ids = td["ids"]
        else:
            # Default IDs for batching
            bs = td.batch_size[0] if len(td.batch_size) > 0 else 1
            self.ids = torch.arange(bs, device=td.device).unsqueeze(-1)

        # Expose common properties directly
        self.dist_matrix = td.get("dist", None)

        # Handle 'waste_with_depot' for WCVRP partial updates (now standardized to 'waste')
        self.waste_with_depot = td.get("waste")

    def get_mask(self) -> Optional[torch.Tensor]:
        """Get action mask from TensorDict."""
        # RL4CO envs provide "action_mask" where True means VALID.
        # AttentionDecoder expects mask where True means INVALID (masked out).
        # So we invert it.
        if "action_mask" in self.td.keys():
            mask = ~self.td["action_mask"]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            return mask
        return None

    def get_edges_mask(self) -> Optional[torch.Tensor]:
        """Get edge mask from TensorDict."""
        return self.td.get("graph_mask", None)

    def get_current_node(self) -> torch.Tensor:
        """Get current node (last visited)."""
        return self.td["current_node"].long()

    def get_current_profit(self) -> torch.Tensor:
        """For VRPP: get cumulative collected waste."""
        # This is used for context embedding.
        val = self.td.get("collected_waste", torch.zeros(self.td.batch_size, device=self.td.device))
        if val.dim() == 1:
            val = val.unsqueeze(-1)
        return val

    def get_current_efficiency(self) -> torch.Tensor:
        """For WCVRP: get current efficiency."""
        # Legacy placeholder
        val = torch.zeros(self.td.batch_size, device=self.td.device)
        return val.unsqueeze(-1)

    def get_remaining_overflows(self) -> torch.Tensor:
        """For WCVRP: get remaining overflows."""
        # Legacy placeholder or value from td
        val = self.td.get("remaining_overflows", torch.zeros(self.td.batch_size, device=self.td.device))
        if val.dim() == 1:
            val = val.unsqueeze(-1)
        return val

    def update(self, action: torch.Tensor) -> "TensorDictStateWrapper":
        """Update state by taking an action in the environment."""
        if self.env is None:
            raise ValueError("Environment (env) must be provided to TensorDictStateWrapper for updates.")

        # Prepare action for environment
        self.td["action"] = action
        next_td = self.env.step(self.td)["next"]
        return TensorDictStateWrapper(next_td, self.problem_name, self.env)

    def all_finished(self) -> bool:
        """Check if all instances in batch are finished."""
        if "done" in self.td.keys():
            return self.td["done"].all().item()
        return False

    def get_finished(self) -> torch.Tensor:
        """Get finished mask for the batch."""
        return self.td.get("done", torch.zeros(self.td.batch_size, dtype=torch.bool, device=self.td.device))

    def __getitem__(self, key):
        """Allow slicing or indexing the state."""
        return TensorDictStateWrapper(self.td[key], self.problem_name, self.env)
