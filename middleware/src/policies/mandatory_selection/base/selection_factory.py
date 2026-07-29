"""
Selection Factory Module.

This module implements the Factory pattern for creating `MandatorySelectionStrategy`
instances. It allows creating strategies by name or from a configuration object.

Attributes:
    None

Example:
    >>> from logic.src.policies.mandatory.base.selection_factory import MandatorySelectionFactory
    >>> strategy = MandatorySelectionFactory.create_strategy("regular", threshold=2)
"""

from typing import Any, Optional, Type, cast

from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy

from .selection_registry import MandatorySelectionRegistry

# Strategy to sub-config attribute mapping in MandatorySelectionConfig
CONFIG_MAPPING = {
    "last_minute": "last_minute",
    "regular": "regular",
    "service_level": "service_level",
    "revenue": "revenue",
    "revenue_threshold": "revenue",
    "lookahead": "lookahead",
    "deadline": "deadline",
    "multi_day_prob": "multi_day_prob",
    "pareto_front": "pareto_front",
    "profit_per_km": "profit_per_km",
    "spatial_synergy": "spatial_synergy",
    "stochastic_regret": "stochastic_regret",
    "combined": "combined",
    "manager": "manager",
    "mip_knapsack": "mip_knapsack",
    "fractional_knapsack": "fractional_knapsack",
    "rollout": "rollout",
    "whittle": "whittle",
    "cvar": "cvar",
    "savings": "savings",
    "set_cover": "set_cover",
    "submodular_greedy": "modular_greedy",
    "supermodular_greedy": "modular_greedy",
    "greedy_routing_heuristic": "modular_greedy",
    "learned": "learned",
    "wasserstein_robust": "wasserstein",
    "dispatcher_thompson": "thompson_dispatcher",
    "dispatcher_portfolio": "thompson_dispatcher",
    "lagrangian": "lagrangian",
    "filter_and_fan": "filter_and_fan",
}


class MandatorySelectionFactory:
    """Factory for creating Mandatory selection strategies."""

    @staticmethod
    def create_strategy(name: str, **kwargs) -> IMandatorySelectionStrategy:
        """
        Create a selection strategy by name.

        Args:
            name: Name of the strategy.
            **kwargs: Arguments to pass to the strategy constructor.
        """
        # Lazy imports to avoid circular dependencies and keep strategies separated
        from logic.src.policies.mandatory_selection.selection_combined import CombinedSelection
        from logic.src.policies.mandatory_selection.selection_cvar import CVaRSelection
        from logic.src.policies.mandatory_selection.selection_deadline import DeadlineDrivenSelection
        from logic.src.policies.mandatory_selection.selection_dispatcher_portfolio import PortfolioDispatcher
        from logic.src.policies.mandatory_selection.selection_dispatcher_thompson import ThompsonDispatcher
        from logic.src.policies.mandatory_selection.selection_filter_and_fan import FilterAndFanSelection
        from logic.src.policies.mandatory_selection.selection_fractional_knapsack import FractionalKnapsackSelection
        from logic.src.policies.mandatory_selection.selection_lagrangian import LagrangianSelection
        from logic.src.policies.mandatory_selection.selection_last_minute import LastMinuteSelection
        from logic.src.policies.mandatory_selection.selection_learned import LearnedSelection
        from logic.src.policies.mandatory_selection.selection_lookahead import LookaheadSelection
        from logic.src.policies.mandatory_selection.selection_mip_knapsack import MIPKnapsackSelection
        from logic.src.policies.mandatory_selection.selection_multi_day_prob import MultiDayOverflowSelection
        from logic.src.policies.mandatory_selection.selection_pareto import ParetoFrontSelection
        from logic.src.policies.mandatory_selection.selection_profit_per_km import ProfitPerKmSelection
        from logic.src.policies.mandatory_selection.selection_regular import RegularSelection
        from logic.src.policies.mandatory_selection.selection_revenue import RevenueThresholdSelection
        from logic.src.policies.mandatory_selection.selection_rollout import RolloutSelection
        from logic.src.policies.mandatory_selection.selection_savings import SavingsSelection
        from logic.src.policies.mandatory_selection.selection_service_level import ServiceLevelSelection
        from logic.src.policies.mandatory_selection.selection_set_cover import SetCoverSelection
        from logic.src.policies.mandatory_selection.selection_spatial_synergy import SpatialSynergySelection
        from logic.src.policies.mandatory_selection.selection_stochastic_regret import StochasticRegretSelection
        from logic.src.policies.mandatory_selection.selection_submodular_greedy import SubmodularGreedySelection
        from logic.src.policies.mandatory_selection.selection_supermodular_greedy import SupermodularGreedySelection
        from logic.src.policies.mandatory_selection.selection_wasserstein import WassersteinRobustSelection
        from logic.src.policies.mandatory_selection.selection_whittle import WhittleIndexSelection

        default_map = {
            "service_level": ServiceLevelSelection,
            "regular": RegularSelection,
            "lookahead": LookaheadSelection,
            "last_minute": LastMinuteSelection,
            "revenue": RevenueThresholdSelection,
            "revenue_threshold": RevenueThresholdSelection,
            "mip_knapsack": MIPKnapsackSelection,
            "fractional_knapsack": FractionalKnapsackSelection,
            "combined": CombinedSelection,
            "deadline": DeadlineDrivenSelection,
            "multi_day_prob": MultiDayOverflowSelection,
            "pareto_front": ParetoFrontSelection,
            "profit_per_km": ProfitPerKmSelection,
            "spatial_synergy": SpatialSynergySelection,
            "stochastic_regret": StochasticRegretSelection,
            "lagrangian": LagrangianSelection,
            "rollout": RolloutSelection,
            "whittle": WhittleIndexSelection,
            "cvar": CVaRSelection,
            "savings": SavingsSelection,
            "set_cover": SetCoverSelection,
            "submodular_greedy": SubmodularGreedySelection,
            "supermodular_greedy": SupermodularGreedySelection,
            "greedy_routing_heuristic": SupermodularGreedySelection,
            "learned": LearnedSelection,
            "wasserstein_robust": WassersteinRobustSelection,
            "dispatcher_thompson": ThompsonDispatcher,
            "dispatcher_portfolio": PortfolioDispatcher,
            "filter_and_fan": FilterAndFanSelection,
        }

        # Check explicit registry first
        cls = MandatorySelectionRegistry.get_strategy_class(name)
        if cls:
            # If strategy accepts kwargs, pass them.
            try:
                return cls(**kwargs)
            except TypeError:
                # Fallback for strategies that don't accept args
                return cls()

        # Check default map
        cls_def = cast(Optional[Type[IMandatorySelectionStrategy]], default_map.get(name.lower()))
        if cls_def:
            try:
                return cls_def(**kwargs)
            except TypeError:
                # Fallback for strategies that don't accept args
                return cls_def()

        raise ValueError(f"Unknown selection strategy: {name}")

    @classmethod
    def create_from_config(cls, config: Any) -> IMandatorySelectionStrategy:
        """
        Create a selection strategy from a MandatorySelectionConfig object.

        Args:
            config: MandatorySelectionConfig instance.
        """
        if config.strategy is None:
            raise ValueError("No strategy specified in MandatorySelectionConfig")

        strategy_name = config.strategy.lower()
        params = config.params.copy() if hasattr(config, "params") else {}

        attr_name = CONFIG_MAPPING.get(strategy_name)
        if attr_name and hasattr(config, attr_name):
            sub_config = getattr(config, attr_name)
            # Convert dataclass to dict for kwargs
            from dataclasses import asdict

            # Check if sub_config is indeed a dataclass (it should be in structured config)
            if hasattr(sub_config, "__dataclass_fields__"):
                sub_params = asdict(sub_config)
                # Some strategies use 'threshold' as 'frequency' or 'confidence_factor'
                # but we've standardized the sub-configs to use the correct names.
                # However, the strategy __init__ might still expect 'threshold'.
                # We'll map them here if necessary.
                if strategy_name == "regular" and "frequency" in sub_params:
                    sub_params["threshold"] = sub_params.pop("frequency")
                elif strategy_name == "service_level" and "confidence_factor" in sub_params:
                    sub_params["threshold"] = sub_params.pop("confidence_factor")
                elif strategy_name == "learned" and "learned_threshold" in sub_params:
                    sub_params["threshold"] = sub_params.pop("learned_threshold")

                params.update(sub_params)
            elif isinstance(sub_config, dict):
                params.update(sub_config)

        # Special handling for CombinedSelection which needs the list of configs
        if strategy_name == "combined" and "combined_strategies" not in params and "strategies" in params:
            # If it's the new CombinedSelectionConfig, it uses 'strategies'
            params["combined_strategies"] = params.pop("strategies")

        return cls.create_strategy(config.strategy, **params)
