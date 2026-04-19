from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class OnlyImprovingConfig:
    """Configuration for Only Improving acceptance."""

    pass


@dataclass
class ImprovingAndEqualConfig:
    """Configuration for Improving and Equal acceptance."""

    pass


@dataclass
class AllMovesAcceptedConfig:
    """Configuration for All Moves Accepted (random walk)."""

    pass


@dataclass
class BoltzmannAcceptanceConfig:
    """Configuration for Boltzmann-Metropolis (Simulated Annealing)."""

    initial_temp: float = 100.0
    alpha: float = 0.95
    seed: int = 42


@dataclass
class DemonAlgorithmConfig:
    """Configuration for Demon Algorithm."""

    initial_credit: float = 0.0
    is_stochastic: bool = False
    max_demon_credit: Optional[float] = None
    maximization: bool = True
    seed: int = 42


@dataclass
class GeneralizedTsallisSAConfig:
    """Configuration for Generalized Tsallis SA."""

    q: float = 1.0
    initial_temp: float = 100.0
    alpha: float = 0.95
    seed: int = 42
    maximization: bool = True


@dataclass
class NonLinearGreatDelugeConfig:
    """Configuration for Non-Linear Great Deluge."""

    initial_level: float = 100.0
    beta: float = 0.01
    t_max: int = 1000
    maximization: bool = True


@dataclass
class EMCQConfig:
    """Configuration for Exponential Monte Carlo with Counter."""

    p: float = 0.1
    p_boost: float = 0.5
    q_threshold: int = 50
    seed: int = 42


@dataclass
class AdaptiveBoltzmannConfig:
    """Configuration for Adaptive Boltzmann Metropolis."""

    p0: float = 0.5
    window_size: int = 100
    alpha: float = 0.95
    min_temp: float = 1e-6
    seed: int = 42
    maximization: bool = True


@dataclass
class LateAcceptanceConfig:
    """Configuration for Late Acceptance Hill Climbing."""

    history_length: int = 50
    maximization: bool = True


@dataclass
class StepCountingConfig:
    """Configuration for Step Counting Hill Climbing."""

    step_limit: int = 100
    maximization: bool = True


@dataclass
class GreatDelugeConfig:
    """Configuration for Great Deluge."""

    initial_level: float = 100.0
    decay_rate: float = 0.01
    maximization: bool = True


@dataclass
class ThresholdAcceptingConfig:
    """Configuration for Threshold Accepting."""

    initial_threshold: float = 100.0
    decay_rate: float = 0.95
    maximization: bool = True


@dataclass
class MonteCarloConfig:
    """Configuration for Fixed Monte Carlo."""

    p: float = 0.1
    seed: int = 42


@dataclass
class AspirationConfig:
    """Configuration for Aspiration Criterion."""

    pass


@dataclass
class AcceptanceConfig:
    """
    Unified configuration wrapper for selecting an acceptance criterion.

    Attributes:
        method: The identifier for the criterion (e.g., 'boltzmann', 'demon').
        params: The specific configuration object for the chosen method.
    """

    method: str = "only_improving"
    params: Any = field(default_factory=OnlyImprovingConfig)
