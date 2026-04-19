from dataclasses import dataclass


@dataclass
class ExactSDPConfig:
    """Configuration for the Exact Stochastic Dynamic Programming solver.

    Attributes:
        time_limit: Max time allowed for the backward induction (mainly COP setup).
        num_days: The horizon limit D for the backward induction.
        discrete_levels: Number of integer discretization levels for the fill levels.
        max_fill_rate: Expected max total probability growth to discretize.
        max_nodes: Hard limit on number of nodes (default 8) to prevent memory overflow.
        discount_factor: Temporal discount for future rewards (gamma).
        overflow_penalty: Cost per unit of overflowing discrete fill level.
        cost_weight: Weight applied to travel distances.
        waste_weight: Reward applied per unit of collected discrete load.
    """

    time_limit: float = 3600.0
    num_days: int = 5
    discrete_levels: int = 4
    max_fill_rate: float = 0.3
    max_nodes: int = 8
    discount_factor: float = 0.99
    overflow_penalty: float = 100.0
    cost_weight: float = 1.0
    waste_weight: float = 10.0
