from dataclasses import MISSING, dataclass, fields
from typing import Any


@dataclass
class LNSMIPParams:
    k_destroy: int = 10
    d_destroy: int = 3
    max_iterations: int = 200
    mip_time_limit: float = 10.0
    mip_gap: float = 0.01
    acceptance: str = "improving"  # "improving" | "sa"
    sa_temperature: float = 100.0
    sa_cooling: float = 0.97
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> "LNSMIPParams":
        params = {}
        for f in fields(cls):
            val = getattr(config, f.name, f.default)
            # Handle MISSING from dataclasses fields or OmegaConf
            if str(val) == "MISSING" or val is MISSING:
                continue
            params[f.name] = val
        return cls(**params)
