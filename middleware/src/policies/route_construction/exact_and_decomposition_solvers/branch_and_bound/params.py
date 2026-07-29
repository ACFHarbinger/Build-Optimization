"""
Parameter dataclasses for Branch-and-Bound solvers.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BBParams:
    """
    Standardized parameters for Branch-and-Bound solvers (MTZ, DFJ, and LR-UOP).

    Shared parameters apply to all three formulations. Parameters prefixed with
    ``lr_`` are only used when ``formulation = "lr_uop"``.
    """

    # Core Solver Parameters
    time_limit: float = 60.0
    mip_gap: float = 0.01
    seed: int = 42

    # MTZ Specific Parameters
    branching_strategy: str = "strong"  # "strong", "most_fractional", "least_fractional"
    strong_branching_limit: int = 5

    # Formulation
    formulation: str = "dfj"  # "dfj", "mtz", or "lr_uop"

    # LR-UOP: Subgradient phase
    lr_lambda_init: float = 0.0  # Initial Lagrange multiplier λ₀ ≥ 0
    lr_max_subgradient_iters: int = 100  # Maximum Polyak iterations
    lr_subgradient_theta: float = 1.0  # Step-size multiplier θ ∈ (0, 2]
    lr_subgradient_time_fraction: float = 0.4  # Fraction of time_limit for Phase 1

    # LR-UOP: Inner uncapacitated OP solver
    lr_op_time_limit: float = 10.0  # Per-solve time limit for the OP (seconds)

    # LR-UOP: B&B phase
    lr_branching_strategy: str = "max_waste"  # "max_waste" or "min_profit"
    lr_max_bb_nodes: int = 5000  # B&B node limit (safety backstop)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BBParams":
        """
        Create a BBParams instance from a raw configuration dictionary.

        Args:
            config: Dictionary containing parameter overrides.

        Returns:
            A BBParams instance with values mapped from the config.
        """
        return cls(
            time_limit=float(config.get("time_limit", 60.0)),
            mip_gap=float(config.get("mip_gap", 0.01)),
            seed=int(config.get("seed", 42)),
            branching_strategy=str(config.get("branching_strategy", "strong")),
            strong_branching_limit=int(config.get("strong_branching_limit", 5)),
            formulation=str(config.get("formulation", "dfj")),
            # LR-UOP parameters
            lr_lambda_init=float(config.get("lr_lambda_init", 0.0)),
            lr_max_subgradient_iters=int(config.get("lr_max_subgradient_iters", 100)),
            lr_subgradient_theta=float(config.get("lr_subgradient_theta", 1.0)),
            lr_subgradient_time_fraction=float(config.get("lr_subgradient_time_fraction", 0.4)),
            lr_op_time_limit=float(config.get("lr_op_time_limit", 10.0)),
            lr_branching_strategy=str(config.get("lr_branching_strategy", "max_waste")),
            lr_max_bb_nodes=int(config.get("lr_max_bb_nodes", 5000)),
        )
