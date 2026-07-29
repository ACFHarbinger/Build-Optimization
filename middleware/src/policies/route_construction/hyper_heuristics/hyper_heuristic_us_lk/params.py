"""
HULK (Hyper-heuristic with Unstringing/stringing and LK) Parameters.

Reference:
    Müller, L. F., & Bonilha, I. (2022). "Hyper-Heuristic Based on ACO
    and Local Search for Dynamic Optimization Problems."
    Algorithms, 15(1), 9. https://doi.org/10.3390/a15010009
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class HULKParams:
    """Parameters for HULK hyper-heuristic."""

    # Search parameters
    max_iterations: int = 1000
    time_limit: float = 300.0
    restarts: int = 3
    restart_threshold: int = 100
    vrpp: bool = False
    profit_aware_operators: bool = False
    seed: Optional[int] = None

    # Operator selection parameters
    epsilon: float = 0.3
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05

    # Performance tracking
    memory_size: int = 50

    # Acceptance criteria
    accept_worse_prob: float = 0.1
    acceptance_decay: float = 0.99

    # Temperature for simulated annealing
    start_temp: float = 100.0
    cooling_rate: float = 0.99
    min_temp: float = 0.01

    # Unstringing/Stringing parameters
    min_destroy_size: int = 2
    max_destroy_pct: float = 0.3

    # Local search parameters
    local_search_iterations: int = 10
    local_search_operators: List[str] = field(
        default_factory=lambda: [
            "2-opt",
            "3-opt",
            "swap",
            "relocate",
        ]
    )

    # Unstringing operators (destroy)
    unstring_operators: List[str] = field(
        default_factory=lambda: [
            "type_i",
            "type_ii",
            "type_iii",
            "type_iv",
        ]
    )

    # Stringing operators (repair)
    string_operators: List[str] = field(
        default_factory=lambda: [
            "type_i",
            "type_ii",
            "type_iii",
            "type_iv",
        ]
    )

    # Müller & Bonilha (2022) Performance scoring (Alpha, Beta, Gamma, Delta)
    score_alpha: float = 20.0
    score_beta: float = 10.0
    score_gamma: float = 5.0
    score_delta: float = 0.5

    # Aliases
    score_best: float = 20.0
    score_improvement: float = 10.0
    score_accept: float = 5.0
    score_reject: float = 0.5

    # Operator weights
    operator_weights: dict = field(default_factory=dict)
    weight_decay: float = 0.95
    weight_learning_rate: float = 0.1

    @classmethod
    def from_config(cls, config: Any) -> "HULKParams":
        """Create parameters from a configuration object."""
        return cls(
            max_iterations=getattr(config, "max_iterations", 1000),
            time_limit=getattr(config, "time_limit", 300.0),
            restarts=getattr(config, "restarts", 3),
            restart_threshold=getattr(config, "restart_threshold", 100),
            vrpp=getattr(config, "vrpp", False),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            epsilon=getattr(config, "epsilon", 0.3),
            epsilon_decay=getattr(config, "epsilon_decay", 0.995),
            min_epsilon=getattr(config, "min_epsilon", 0.05),
            memory_size=getattr(config, "memory_size", 50),
            accept_worse_prob=getattr(config, "accept_worse_prob", 0.1),
            acceptance_decay=getattr(config, "acceptance_decay", 0.99),
            start_temp=getattr(config, "start_temp", 100.0),
            cooling_rate=getattr(config, "cooling_rate", 0.99),
            min_temp=getattr(config, "min_temp", 0.01),
            min_destroy_size=getattr(config, "min_destroy_size", 2),
            max_destroy_pct=getattr(config, "max_destroy_pct", 0.3),
            local_search_iterations=getattr(config, "local_search_iterations", 10),
            local_search_operators=getattr(config, "local_search_operators", ["2-opt", "3-opt", "swap", "relocate"]),
            unstring_operators=getattr(config, "unstring_operators", ["type_i", "type_ii", "type_iii", "type_iv"]),
            string_operators=getattr(config, "string_operators", ["type_i", "type_ii", "type_iii", "type_iv"]),
            score_alpha=getattr(config, "score_alpha", 10.0),
            score_beta=getattr(config, "score_beta", 5.0),
            score_gamma=getattr(config, "score_gamma", -1.0),
            score_delta=getattr(config, "score_delta", 20.0),
            weight_learning_rate=getattr(config, "weight_learning_rate", 0.1),
            weight_decay=getattr(config, "weight_decay", 0.95),
        )

    def __post_init__(self):
        """Initialize operator weights if not provided."""
        if not self.operator_weights:
            all_ops = self.unstring_operators + self.string_operators + self.local_search_operators
            self.operator_weights = {op: 1.0 for op in all_ops}
