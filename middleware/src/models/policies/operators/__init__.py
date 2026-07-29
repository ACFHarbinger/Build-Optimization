"""
Vectorized local search operators for WSmart-Route.

This module provides GPU-accelerated implementations of classical VRP local search
operators. All operators support batch processing for parallel evaluation across
multiple instances.

Operators are organized into categories:
- Route operators: 2-opt, 3-opt, 2-opt*, swap*
- Move operators: relocate, swap, or-opt
- Exchange operators: cross-exchange, λ-interchange, ejection chain
- Destroy operators: random, worst, cluster, shaw, string removal
- Repair operators: greedy insertion, regret-k insertion
- Unstringing operators: Type I, II, III, IV (sophisticated k-opt moves)
- Advanced operators: LKH (Lin-Kernighan-Helsgaun)
"""

# Destroy operators
from .destroy.cluster_removal import vectorized_cluster_removal
from .destroy.random_removal import vectorized_random_removal
from .destroy.shaw_removal import vectorized_shaw_removal
from .destroy.string_removal import vectorized_string_removal
from .destroy.worst_removal import vectorized_worst_removal

# Exchange operators
from .exchange.cross_exchange import vectorized_cross_exchange
from .exchange.ejection_chain import vectorized_ejection_chain
from .exchange.lambda_interchange import vectorized_lambda_interchange
from .exchange.or_opt import vectorized_or_opt

# Move operators
from .move.relocate import vectorized_relocate
from .move.swap import vectorized_swap

# Repair operators
from .repair.greedy_insertion import vectorized_greedy_insertion
from .repair.regret_k_insertion import vectorized_regret_k_insertion

# Route operators
from .route.lkh import vectorized_lkh
from .route.swap_star import vectorized_swap_star
from .route.three_opt import vectorized_three_opt
from .route.two_opt import vectorized_two_opt
from .route.two_opt_star import vectorized_two_opt_star

# Unstringing operators
from .unstringing import (
    vectorized_type_i_unstringing,
    vectorized_type_ii_unstringing,
    vectorized_type_iii_unstringing,
    vectorized_type_iv_unstringing,
)

__all__ = [
    # Route operators
    "vectorized_two_opt",
    "vectorized_three_opt",
    "vectorized_two_opt_star",
    "vectorized_swap_star",
    "vectorized_lkh",
    # Move operators
    "vectorized_swap",
    "vectorized_relocate",
    # Exchange operators
    "vectorized_or_opt",
    "vectorized_cross_exchange",
    "vectorized_lambda_interchange",
    "vectorized_ejection_chain",
    # Destroy operators
    "vectorized_random_removal",
    "vectorized_worst_removal",
    "vectorized_cluster_removal",
    "vectorized_shaw_removal",
    "vectorized_string_removal",
    # Repair operators
    "vectorized_greedy_insertion",
    "vectorized_regret_k_insertion",
    # Unstringing operators
    "vectorized_type_i_unstringing",
    "vectorized_type_ii_unstringing",
    "vectorized_type_iii_unstringing",
    "vectorized_type_iv_unstringing",
]
