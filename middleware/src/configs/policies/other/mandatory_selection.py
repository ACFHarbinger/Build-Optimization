"""
Mandatory Selection Config module.

Configures the strategies used to determine which bins must be collected
during simulation or training. This module follows the modular pattern of
nested dataclasses for each algorithm.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LastMinuteSelectionConfig:
    """Configuration for threshold-based last-minute selection."""

    threshold: float = 0.7


@dataclass
class RegularSelectionConfig:
    """Configuration for periodic collection on scheduled days."""

    frequency: int = 3


@dataclass
class ServiceLevelSelectionConfig:
    """Configuration for statistical overflow prediction."""

    confidence_factor: float = 1.0


@dataclass
class RevenueSelectionConfig:
    """Configuration for revenue-based selection."""

    revenue_kg: float = 1.0
    bin_capacity: float = 1.0
    revenue_threshold: float = 0.0


@dataclass
class LookaheadSelectionConfig:
    """Configuration for predictive collection within a horizon."""

    horizon_days: int = 3
    threshold: float = 0.7  # specific usage depends on implementation


@dataclass
class DeadlineSelectionConfig:
    """Configuration for deterministic days-to-overflow selection."""

    horizon_days: int = 3
    threshold: float = 0.9  # used as urgency threshold


@dataclass
class MultiDayProbSelectionConfig:
    """Configuration for stochastic overflow risk over K days."""

    horizon_days: int = 3
    threshold: float = 0.5  # risk probability threshold


@dataclass
class ParetoFrontSelectionConfig:
    """Configuration for multi-objective (urgency x distance) optimization."""

    threshold: float = 0.5


@dataclass
class ProfitPerKmSelectionConfig:
    """Configuration for spatial ROI (Expected revenue / distance)."""

    threshold: float = 0.0
    revenue_kg: float = 1.0


@dataclass
class SpatialSynergySelectionConfig:
    """Configuration for critical bins + opportunistic neighbors."""

    critical_threshold: float = 0.90
    synergy_threshold: float = 0.60
    radius: float = 10.0


@dataclass
class StochasticRegretSelectionConfig:
    """Configuration for expected overflow volume minimization."""

    threshold: float = 0.1


@dataclass
class CombinedSelectionConfig:
    """Configuration for combining multiple strategies with OR/AND logic."""

    strategies: Optional[List[Dict[str, Any]]] = None
    logic: str = "or"


@dataclass
class MandatoryManagerSelectionConfig:
    """Configuration for neural network-based selection (MandatoryManager)."""

    hidden_dim: int = 128
    lstm_hidden: int = 64
    history_length: int = 10
    manager_critical_threshold: float = 0.9
    manager_weights: Optional[str] = None
    device: str = "cuda"


@dataclass
class KnapsackSelectionConfig:
    """Base configuration for knapsack-based economic coupling."""

    n_vehicles: int = 1
    cost_per_km: float = 0.1
    use_eoq_threshold: bool = False
    holding_cost_per_kg_day: float = 0.0
    ordering_cost_per_visit: float = 0.0


@dataclass
class MIPKnapsackSelectionConfig(KnapsackSelectionConfig):
    """Configuration for exact 0/1 multiple-knapsack MIP selection."""

    overflow_penalty_frac: float = 1.0  # additional penalty for overflow event


@dataclass
class FractionalKnapsackSelectionConfig(KnapsackSelectionConfig):
    """Configuration for greedy net-profit density selection."""


@dataclass
class RolloutSelectionConfig:
    """Configuration for rollout-based predictive selection."""

    rollout_horizon: int = 5
    rollout_base_policy: str = "last_minute"
    rollout_n_scenarios: int = 1
    rollout_discount: float = 0.95


@dataclass
class WhittleSelectionConfig:
    """Configuration for Whittle index-based allocation."""

    whittle_discount: float = 0.95
    whittle_grid_size: int = 21
    n_vehicles: int = 1


@dataclass
class CVaRSelectionConfig:
    """Configuration for Conditional Value at Risk selection."""

    cvar_alpha: float = 0.95
    threshold: float = 0.0


@dataclass
class SavingsSelectionConfig:
    """Configuration for savings-based (Clarke-Wright) pre-selection."""

    savings_min_fill_ratio: float = 0.5


@dataclass
class SetCoverSelectionConfig:
    """Configuration for set-covering based selection."""

    service_radius: float = 5.0
    critical_threshold: float = 0.90


@dataclass
class ModularGreedySelectionConfig:
    """Configuration for (Super/Sub-)Modular greedy selection."""

    modular_alpha: float = 1.0
    modular_budget: int = 0


@dataclass
class LearnedSelectionConfig:
    """Configuration for imitation/learned selection models."""

    learned_model_path: Optional[str] = None
    learned_threshold: float = 0.5


@dataclass
class ThompsonDispatcherSelectionConfig:
    """Configuration for contextual Thompson sampling dispatcher."""

    dispatcher_state_path: Optional[str] = None
    dispatcher_candidate_strategies: Optional[List[str]] = None
    dispatcher_exploration: float = 1.0
    dispatcher_mode: str = "union"


@dataclass
class WassersteinSelectionConfig:
    """Configuration for distributionally robust Wasserstein selection."""

    wasserstein_radius: float = 0.1
    wasserstein_p: int = 1


@dataclass
class LagrangianSelectionConfig:
    """Configuration for Lagrangian relaxation based selection."""

    n_vehicles: int = 1
    cost_per_km: float = 0.1


@dataclass
class MandatorySelectionConfig:
    """Main configuration for mandatory bin selection.

    Composes algorithm-specific parameters and execution settings.

    Attributes:
        strategy: Primary selection strategy name.
        max_fill: Global overflow threshold (0.0 to 1.0 or 100.0).
        last_minute: Params for LastMinuteSelection.
        regular: Params for RegularSelection.
        service_level: Params for ServiceLevelSelection.
        revenue: Params for RevenueSelection.
        lookahead: Params for LookaheadSelection.
        deadline: Params for DeadlineDrivenSelection.
        multi_day_prob: Params for MultiDayOverflowSelection.
        pareto_front: Params for ParetoFrontSelection.
        profit_per_km: Params for ProfitPerKmSelection.
        spatial_synergy: Params for SpatialSynergySelection.
        stochastic_regret: Params for StochasticRegretSelection.
        combined: Params for CombinedSelection.
        manager: Params for neural MandatoryManager.
        mip_knapsack: Params for MIPKnapsackSelection.
        fractional_knapsack: Params for FractionalKnapsackSelection.
        rollout: Params for RolloutSelection.
        whittle: Params for WhittleIndexSelection.
        cvar: Params for CVaRSelection.
        savings: Params for SavingsSelection.
        set_cover: Params for SetCoverSelection.
        modular_greedy: Params for ModularGreedySelection.
        learned: Params for LearnedSelection.
        thompson_dispatcher: Params for ThompsonDispatcher.
        wasserstein: Params for WassersteinSelection.
        lagrangian: Params for LagrangianSelection.
        params: Additional strategy-specific parameters as a dictionary.
    """

    strategy: Optional[str] = None
    max_fill: float = 1.0

    # Strategy-specific sub-configs
    last_minute: LastMinuteSelectionConfig = field(default_factory=LastMinuteSelectionConfig)
    regular: RegularSelectionConfig = field(default_factory=RegularSelectionConfig)
    service_level: ServiceLevelSelectionConfig = field(default_factory=ServiceLevelSelectionConfig)
    revenue: RevenueSelectionConfig = field(default_factory=RevenueSelectionConfig)
    lookahead: LookaheadSelectionConfig = field(default_factory=LookaheadSelectionConfig)
    deadline: DeadlineSelectionConfig = field(default_factory=DeadlineSelectionConfig)
    multi_day_prob: MultiDayProbSelectionConfig = field(default_factory=MultiDayProbSelectionConfig)
    pareto_front: ParetoFrontSelectionConfig = field(default_factory=ParetoFrontSelectionConfig)
    profit_per_km: ProfitPerKmSelectionConfig = field(default_factory=ProfitPerKmSelectionConfig)
    spatial_synergy: SpatialSynergySelectionConfig = field(default_factory=SpatialSynergySelectionConfig)
    stochastic_regret: StochasticRegretSelectionConfig = field(default_factory=StochasticRegretSelectionConfig)
    combined: CombinedSelectionConfig = field(default_factory=CombinedSelectionConfig)
    manager: MandatoryManagerSelectionConfig = field(default_factory=MandatoryManagerSelectionConfig)
    mip_knapsack: MIPKnapsackSelectionConfig = field(default_factory=MIPKnapsackSelectionConfig)
    fractional_knapsack: FractionalKnapsackSelectionConfig = field(default_factory=FractionalKnapsackSelectionConfig)
    rollout: RolloutSelectionConfig = field(default_factory=RolloutSelectionConfig)
    whittle: WhittleSelectionConfig = field(default_factory=WhittleSelectionConfig)
    cvar: CVaRSelectionConfig = field(default_factory=CVaRSelectionConfig)
    savings: SavingsSelectionConfig = field(default_factory=SavingsSelectionConfig)
    set_cover: SetCoverSelectionConfig = field(default_factory=SetCoverSelectionConfig)
    modular_greedy: ModularGreedySelectionConfig = field(default_factory=ModularGreedySelectionConfig)
    learned: LearnedSelectionConfig = field(default_factory=LearnedSelectionConfig)
    thompson_dispatcher: ThompsonDispatcherSelectionConfig = field(default_factory=ThompsonDispatcherSelectionConfig)
    wasserstein: WassersteinSelectionConfig = field(default_factory=WassersteinSelectionConfig)
    lagrangian: LagrangianSelectionConfig = field(default_factory=LagrangianSelectionConfig)

    # Additional parameters
    params: Dict[str, Any] = field(default_factory=dict)
