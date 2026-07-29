"""
Normalization configuration module.
"""

from dataclasses import dataclass

NORM_EPSILON: float = 1e-5


@dataclass
class NormalizationConfig:
    """Configuration for normalization layers."""

    norm_type: str = "batch"
    epsilon: float = NORM_EPSILON
    learn_affine: bool = True
    track_stats: bool = False
    momentum: float = 0.1
    n_groups: int = 1
    k_lrnorm: float = 1.0
