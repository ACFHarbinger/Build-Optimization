"""
Selection Strategy Implementations for WSmart-Route.

This package contains concrete implementations of the `MandatorySelectionStrategy`
interface. Strategies determine which bins must be collected on a given day
based on different criteria (fill levels, revenue, schedule, etc.).

Includes both:
- Single-instance selectors (for simulation): SelectionContext-based
- Vectorized selectors (for training): Batched PyTorch tensor operations

Attributes:
    CombinedSelector (class): Combines multiple selection strategies.
    LastMinuteSelector (class): Selects bins that are about to overflow.
    LookaheadSelector (class): Selects bins based on future fill predictions.
    ManagerSelector (class): Selects bins based on a manager agent's policy.
    RegularSelector (class): Selects bins based on fixed schedule frequency.
    RevenueSelector (class): Selects bins based on revenue potential.
    ServiceLevelSelector (class): Selects bins to maintain a service level.
    VectorizedSelector (class): Abstract base for vectorized selection.
    create_selector_from_config (function): Factory function for config-based creation.
    get_vectorized_selector (function): Factory function for vectorized selectors.

Example:
    >>> from logic.src.policies.mandatory import create_selector_from_config
    >>> selector = create_selector_from_config(config)
    >>> selected_nodes = selector.select(context)
"""

from logic.src.interfaces import IMandatorySelectionStrategy
from logic.src.models.policies.selection import (
    CombinedSelector,
    LastMinuteSelector,
    LookaheadSelector,
    ManagerSelector,
    RegularSelector,
    RevenueSelector,
    ServiceLevelSelector,
    VectorizedSelector,
    create_selector_from_config,
    get_vectorized_selector,
)

from .base import MandatorySelectionFactory, MandatorySelectionRegistry, SelectionContext
from .selection_combined import CombinedSelection
from .selection_cvar import CVaRSelection
from .selection_deadline import DeadlineDrivenSelection
from .selection_dispatcher_portfolio import PortfolioDispatcher
from .selection_dispatcher_thompson import ThompsonDispatcher
from .selection_fractional_knapsack import FractionalKnapsackSelection
from .selection_lagrangian import LagrangianSelection
from .selection_last_minute import LastMinuteSelection
from .selection_learned import LearnedSelection
from .selection_lookahead import LookaheadSelection
from .selection_mip_knapsack import MIPKnapsackSelection
from .selection_multi_day_prob import MultiDayOverflowSelection
from .selection_pareto import ParetoFrontSelection
from .selection_profit_per_km import ProfitPerKmSelection
from .selection_regular import RegularSelection
from .selection_revenue import RevenueThresholdSelection
from .selection_rollout import RolloutSelection
from .selection_savings import SavingsSelection
from .selection_service_level import ServiceLevelSelection
from .selection_set_cover import SetCoverSelection
from .selection_spatial_synergy import SpatialSynergySelection
from .selection_stochastic_regret import StochasticRegretSelection
from .selection_submodular_greedy import SubmodularGreedySelection
from .selection_supermodular_greedy import SupermodularGreedySelection
from .selection_wasserstein import WassersteinRobustSelection
from .selection_whittle import WhittleIndexSelection

__all__ = [
    # Vectorized selectors for training
    "VectorizedSelector",
    "LastMinuteSelector",
    "RegularSelector",
    "LookaheadSelector",
    "RevenueSelector",
    "ServiceLevelSelector",
    "CombinedSelector",
    "ManagerSelector",
    "get_vectorized_selector",
    "create_selector_from_config",
    "IMandatorySelectionStrategy",
    "MandatorySelectionFactory",
    "MandatorySelectionRegistry",
    "SelectionContext",
    # New simulation strategies
    "CombinedSelection",
    "LastMinuteSelection",
    "RevenueThresholdSelection",
    "DeadlineDrivenSelection",
    "MultiDayOverflowSelection",
    "ParetoFrontSelection",
    "ProfitPerKmSelection",
    "SpatialSynergySelection",
    "StochasticRegretSelection",
    "LagrangianSelection",
    "RolloutSelection",
    "WhittleIndexSelection",
    "CVaRSelection",
    "SavingsSelection",
    "SetCoverSelection",
    "SubmodularGreedySelection",
    "SupermodularGreedySelection",
    "LearnedSelection",
    "WassersteinRobustSelection",
    "ThompsonDispatcher",
    "PortfolioDispatcher",
    "RegularSelection",
    "LookaheadSelection",
    "ServiceLevelSelection",
    "FractionalKnapsackSelection",
    "MIPKnapsackSelection",
]
