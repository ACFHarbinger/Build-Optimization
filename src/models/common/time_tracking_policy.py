import time
from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase


class TimeTrackingPolicy(nn.Module):
    """
    Wrapper for policies to track inference time.

    injects "inference_time" into the output dictionary.
    """

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, td: TensorDict, env: Optional[RL4COEnvBase] = None, **kwargs) -> Dict[str, Any]:
        # Synchronize for accurate GPU timing
        if td.device is not None and td.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        # Run policy
        out = self.policy(td, env, **kwargs)

        if td.device is not None and td.device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        # Calculate duration
        duration = end - start

        # Inject into output
        # Use same device as reward for mathematical compatibility
        device = out["reward"].device if "reward" in out else td.device

        out["inference_time"] = torch.tensor(duration, dtype=torch.float32, device=device)

        return out

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to underlying policy."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.policy, name)
