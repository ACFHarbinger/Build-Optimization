"""
Configuration parameters for the GENIUS (GENI + US) meta-heuristic.

Reference:
    Gendreau, M., Hertz, A., & Laporte, G. (1992).
    "New Insertion and Postoptimization Procedures for the Traveling Salesman Problem"
    Operations Research, 40(6), 1086-1094.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class GENIUSParams:
    """
    Configuration for the GENIUS meta-heuristic solver.

    GENIUS combines two procedures:
    - **GENI (Generalized Insertion)**: Efficient insertion heuristic that considers
      two types of insertions (Type I and Type II) and uses p-neighborhood restriction.
    - **US (Unstringing and Stringing)**: Post-optimization procedure that removes
      and reinserts vertices to further improve the solution.

    The algorithm follows these steps:
    1. Construct initial solution using GENI insertion
    2. Apply US post-optimization cycles to refine the solution
    3. Iterate until time limit or no improvement

    Attributes:
        neighborhood_size: Size of p-neighborhood for GENI insertion (p parameter).
            Restricts insertion search to p closest vertices to the candidate node.
            Typical values: 3-7. Paper recommends p=5 for good balance.
        unstring_type: Type of unstringing operator to use (1=Type I, 2=Type II,
            3=Type III, 4=Type IV). Each type differs in arc reconnection patterns.
        string_type: Type of stringing operator to use (1=Type I, 2=Type II,
            3=Type III, 4=Type IV). Should match unstring_type for consistency.
        n_iterations: Number of complete GENIUS iterations (GENI + US cycles).
        vrpp: Whether to use VRPP expanded pool (True).
        profit_aware_operators: Whether to use profit-aware operators (True) or
            cost-aware operators (False).
        time_limit: Wall-clock time limit in seconds. Set to 0 for no limit.
        random_us_sampling: Whether to use random sampling (True) or deterministic
            p-neighborhood search (False) for US operators. Default: False for
            strict adherence to Gendreau et al. (1992).
    """

    neighborhood_size: int = 5
    unstring_type: int = 1
    string_type: int = 1
    n_iterations: int = 1
    vrpp: bool = False
    profit_aware_operators: bool = False
    time_limit: float = 60.0
    random_us_sampling: bool = False
    seed: Optional[int] = None

    @classmethod
    def from_config(cls, config: Any) -> GENIUSParams:
        """Build parameters from a configuration object."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in dataclasses.fields(cls)}})

        return cls(
            neighborhood_size=getattr(config, "neighborhood_size", 5),
            unstring_type=getattr(config, "unstring_type", 1),
            string_type=getattr(config, "string_type", 1),
            n_iterations=getattr(config, "n_iterations", 1),
            vrpp=getattr(config, "vrpp", False),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            time_limit=getattr(config, "time_limit", 60.0),
            random_us_sampling=getattr(config, "random_us_sampling", False),
            seed=getattr(config, "seed", None),
        )
