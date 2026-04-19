from typing import Any, Dict, Optional, Union

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion

from .registry import AcceptanceCriterionRegistry


class AcceptanceCriterionFactory:
    """
    Factory for creating move acceptance criteria.

    Integrates with the configuration system to provide solver-specific
    move acceptance logic.
    """

    _registered = False

    @classmethod
    def ensure_registered(cls) -> None:
        """Imports all criteria modules to trigger registration."""
        if cls._registered:
            return

        import contextlib
        import importlib

        # List of modular acceptance criteria implementations to load
        modules = [
            "adaptive_boltzmann_metropolis",
            "all_moves_accepted",
            "aspiration_criterion",
            "binary_tournament_acceptance",
            "boltzmann_metropolis_criterion",
            "demon_algorithm",
            "ensemble_move_acceptance",
            "epsilon_dominance",
            "exponential_monte_carlo_counter",
            "fitness_proportional",
            "generalized_tsallis_simulated_annealing",
            "great_deluge",
            "improving_and_equal",
            "late_acceptance_hill_climbing",
            "monte_carlo",
            "non_linear_great_deluge",
            "old_bachelor_acceptance",
            "only_improving",
            "pareto_dominance",
            "probabilistic_transition",
            "record_to_record_travel",
            "skewed_variable_neighborhood_search",
            "step_counting_hill_climbing",
            "threshold_accepting",
        ]

        for mod in modules:
            with contextlib.suppress(ImportError):
                importlib.import_module(f"logic.src.policies.route_construction.acceptance_criteria.{mod}")

        cls._registered = True

    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[Union[Dict[str, Any], Any]] = None,
        **kwargs: Any,
    ) -> IAcceptanceCriterion:
        """
        Instantiate an acceptance criterion.

        Args:
            name: Identifier for the criterion (e.g., 'boltzmann').
            config: Configuration parameters. Can be a dict or a dataclass instance.
            **kwargs: Fallback parameters if config is not comprehensive.

        Returns:
            Instantiated acceptance criterion.
        """
        cls.ensure_registered()

        criterion_cls = AcceptanceCriterionRegistry.get(name)
        if not criterion_cls:
            raise ValueError(f"Unknown acceptance criterion: {name}")

        # Extract parameters from config if it's a dict or dataclass
        params: Dict[str, Any] = {}
        if config is not None:
            if isinstance(config, dict):
                params = config.copy()
            elif hasattr(config, "__dict__"):
                # Handle dataclasses or simple objects
                params = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}

        # Merge with kwargs (kwargs take precedence for dynamic injection)
        params.update(kwargs)

        # Instantiate. We use filter_kwargs pattern if needed, or rely on constructor signatures.
        # For now, we assume constructors are compatible with the params they receive.
        return criterion_cls(**params)
