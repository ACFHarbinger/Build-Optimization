from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import Any


@dataclass
class TPKSParams:
    # Phase I
    phase1_kernel_size: int = 30  # initial kernel for feasibility search
    phase1_bucket_size: int = 15  # bucket size during Phase I expansion
    phase1_time_fraction: float = 0.35  # fraction of time_limit for Phase I
    phase1_mip_node_limit: int = 2000

    # Phase II
    phase2_bucket_size_easy: int = 30  # larger buckets for EASY instances
    phase2_bucket_size_normal: int = 15
    phase2_time_limit_per_bucket: float = 15.0  # seconds (adaptive override)
    max_buckets: int = 20
    t_easy: float = 8.0  # Phase II difficulty threshold (seconds)
    epsilon: float = 0.1  # HARD: fix variables with relaxation > 1 - ε

    # Shared
    time_limit: float = 300.0
    mip_gap: float = 0.01
    mip_limit_nodes: int = 10000
    initial_kernel_size: int = 50  # fallback if phase1 var stats unavailable
    seed: int = 42
    engine: str = "gurobi"

    @classmethod
    def from_config(cls, config: Any) -> "TPKSParams":
        params = {}
        for f in fields(cls):
            val = getattr(config, f.name, f.default)
            # Handle MISSING from dataclasses fields or OmegaConf
            if str(val) == "MISSING" or val is MISSING:
                continue
            params[f.name] = val
        return cls(**params)

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}
