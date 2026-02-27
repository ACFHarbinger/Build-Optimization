"""
HGS Local Search operators package.

This package provides a comprehensive suite of local search operators for VRP:
- Move operators: relocate, swap
- Route operators: 2-opt, 3-opt, 2-opt*, swap*
- Perturbation operators: perturb, kick
- Destroy operators: random, worst, cluster, shaw, string removal (SISR)
- Repair operators: greedy, regret-2, regret-k, blink insertion (SISR)
- Exchange operators: or-opt, cross-exchange, ejection chain, λ-interchange
"""

# Destroy operators
from .destroy_operators import (
    cluster_removal,
    random_removal,
    shaw_removal,
    string_removal,
    worst_removal,
)

# Exchange operators
from .exchange_operators import (
    cross_exchange,
    ejection_chain,
    lambda_interchange,
    move_or_opt,
)

# Move operators
from .move_operators import move_relocate, move_swap

# Perturbation operators
from .perturbation_operators import kick, perturb

# Repair operators
from .repair_operators import (
    greedy_insertion,
    greedy_insertion_with_blinks,
    regret_2_insertion,
    regret_k_insertion,
)

# Route operators
from .route_operators import (
    move_2opt_intra,
    move_2opt_star,
    move_3opt_intra,
    move_swap_star,
)

# Unstringing operators
from .unstringing_operators import (
    apply_type_i_unstringing,
    apply_type_ii_unstringing,
    apply_type_iii_unstringing,
    apply_type_iv_unstringing,
)

__all__ = [
    # Move
    "move_relocate",
    "move_swap",
    # Route
    "move_2opt_intra",
    "move_2opt_star",
    "move_3opt_intra",
    "move_swap_star",
    # Perturbation
    "perturb",
    "kick",
    # Destroy
    "random_removal",
    "worst_removal",
    "cluster_removal",
    "shaw_removal",
    "string_removal",
    # Repair
    "greedy_insertion",
    "regret_2_insertion",
    "regret_k_insertion",
    "greedy_insertion_with_blinks",
    # Exchange
    "move_or_opt",
    "cross_exchange",
    "ejection_chain",
    "lambda_interchange",
    # Unstringing
    "apply_type_i_unstringing",
    "apply_type_ii_unstringing",
    "apply_type_iii_unstringing",
    "apply_type_iv_unstringing",
]
