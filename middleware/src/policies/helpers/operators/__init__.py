"""
Local Search operators package.

This package provides a comprehensive suite of local search operators for VRP,
organized by category:
- Crossover: GA evolution recombination operators.
- Destroy: Removal heuristics (random, worst, cluster, shaw, string, route,
  neighbor, historical, sector).
- Repair: Insertion heuristics (greedy, regret-2, regret-k, blink, savings, deep).
- Intra-route: Moves within a single route (relocate, swap, 2-opt, 3-opt,
  k-opt, or-opt, geni, k-perm, link-swap, relocate-chain).
- Inter-route: Moves between different routes (swap*, k-opt*, cross, i-cross,
  ejection, lambda, cyclic, relocate (2,0)/(2,1)).
- Perturbation: Kick, perturb, double-bridge, genetic transformation, evolutionary.
- Heuristics: Complex local search heuristics (initialization, LKH).
- Intensification: Steepest-descent local search loops, Held-Karp DP route
  re-optimisation, Fix-and-Optimize sub-MIP, and Set-Partitioning Polish.
- Evolutionary Mutation: Post-crossover GA/DE mutation primitives (swap,
  inversion, scramble, random 2-opt, DE/rand/1, DE/best/1).
- Sequence Merging: Hyper-heuristic operator-sequence construction and
  evolution (ACO, Markov Chain, SS-HH, sequence crossover/mutation).
"""

# Intensification operators
# Crossover operators
from .crossover_recombination import (
    CROSSOVER_NAMES,
    CROSSOVER_OPERATORS,
    capacity_aware_erx,
    edge_recombination_crossover,
    generalized_partition_crossover,
    ordered_crossover,
    random_node_inheritance_crossover,
    route_profit_gpx_crossover,
    selective_route_exchange_crossover,
)

# Destroy operators
from .destroy_ruin import (
    bb_profit_removal,
    bb_removal,
    cluster_profit_removal,
    cluster_removal,
    historical_profit_removal,
    historical_removal,
    neighbor_profit_removal,
    neighbor_removal,
    pattern_removal,
    penalized_removal,
    random_horizon_removal,
    random_removal,
    route_profit_removal,
    route_removal,
    sector_profit_removal,
    sector_removal,
    shaw_horizon_removal,
    shaw_profit_removal,
    shaw_removal,
    shift_visit_removal,
    string_profit_removal,
    string_removal,
    urgency_aware_removal,
    worst_profit_horizon_removal,
    worst_profit_removal,
    worst_removal,
)

# Evolutionary mutation operators
from .evolutionary_mutation import (
    EVOLUTIONARY_MUTATION_NAMES,
    EVOLUTIONARY_MUTATION_OPERATORS,
    de_best_1_mutation,
    de_rand_1_mutation,
    inversion_mutation,
    inversion_mutation_profit,
    random_2opt_mutation,
    random_2opt_mutation_profit,
    scramble_mutation,
    scramble_mutation_profit,
    swap_mutation,
    swap_mutation_profit,
)

# Generalized Insertion and Deletion operators
# Includes Unstringing and Stringing (US)
from .generalized_insertion_and_deletion import (
    apply_type_i_s,
    apply_type_i_s_profit,
    apply_type_i_us,
    apply_type_i_us_profit,
    apply_type_ii_s,
    apply_type_ii_s_profit,
    apply_type_ii_us,
    apply_type_ii_us_profit,
    apply_type_iii_s,
    apply_type_iii_s_profit,
    apply_type_iii_us,
    apply_type_iii_us_profit,
    apply_type_iv_s,
    apply_type_iv_s_profit,
    apply_type_iv_us,
    apply_type_iv_us_profit,
    stringing_insertion,
    stringing_profit_insertion,
    unstringing_profit_removal,
    unstringing_removal,
)

# Improvement operators
from .improvement_descent import (
    node_exchange_steepest,
    node_exchange_steepest_profit,
    or_opt_steepest,
    or_opt_steepest_profit,
    two_opt_steepest,
    two_opt_steepest_profit,
)

# Intensification operators
from .intensification_fixing import (
    INTENSIFICATION_NAMES,
    INTENSIFICATION_OPERATORS,
    dp_route_reopt,
    dp_route_reopt_profit,
    fix_and_optimize,
    fix_and_optimize_profit,
    set_partitioning_polish,
    set_partitioning_polish_profit,
)

# Inter-route operators
from .inter_route_local_search import (
    cross_exchange,
    cyclic_transfer,
    ejection_chain,
    exchange_2_0,
    exchange_2_1,
    exchange_k_0,
    exchange_k_h,
    improved_cross_exchange,
    lambda_interchange,
    move_2opt_star,
    move_3opt_star,
    move_kopt_star,
    move_swap_star,
)

# Intra-route operators
from .intra_route_local_search import (
    apply_intra_route_cross_exchange,
    k_permutation,
    move_2opt_intra,
    move_3opt_intra,
    move_kopt_intra,
    move_or_opt,
    move_relocate,
    move_swap,
    relocate_chain,
    three_permutation,
)

# Perturbation operators
from .perturbation_shaking import (
    bb_perturbation,
    bb_profit_perturbation,
    double_bridge,
    evolutionary_perturbation,
    evolutionary_perturbation_profit,
    genetic_transformation,
    genetic_transformation_profit,
    kick,
    kick_profit,
    perturb,
)

# Repair operators
from .recreate_repair import (
    bb_insertion,
    bb_profit_insertion,
    deep_insertion,
    deep_profit_insertion,
    farthest_insertion,
    forward_looking_insertion,
    geni_insertion,
    geni_profit_insertion,
    greedy_horizon_insertion,
    greedy_insertion,
    greedy_insertion_with_blinks,
    greedy_profit_insertion,
    greedy_profit_insertion_with_blinks,
    nearest_insertion,
    nearest_profit_insertion,
    regret_2_insertion,
    regret_2_profit_insertion,
    regret_3_insertion,
    regret_3_profit_insertion,
    regret_4_insertion,
    regret_4_profit_insertion,
    regret_k_insertion,
    regret_k_profit_insertion,
    regret_k_temporal_insertion,
    savings_insertion,
    savings_profit_insertion,
    stochastic_aware_insertion,
)

# Heuristics
from .search_heuristics import (
    apply_ges,
    apply_lns,
    solve_lk,
    solve_lkh,
)

# Sequence merging (hyper-heuristic operator sequencing)
from .sequence_merging import (
    AcoSequenceState,
    MarkovSequenceState,
    SsHhState,
    aco_best_sequence,
    aco_build_sequence,
    aco_update_pheromones,
    markov_fit_from_log,
    markov_sample_sequence,
    markov_stationary_distribution,
    markov_update,
    sequence_deletion_mutation,
    sequence_insertion_mutation,
    sequence_order_preserving_crossover,
    sequence_single_point_crossover,
    sequence_substitution_mutation,
    sequence_transposition_mutation,
    sequence_uniform_crossover,
    ss_hh_build_sequence,
    ss_hh_decay_scores,
    ss_hh_rank_operators,
    ss_hh_select,
    ss_hh_update,
)

# Solution initialization
from .solution_initialization import (
    build_grasp_routes,
    build_greedy_routes,
    build_nn_routes,
    build_regret_routes,
    build_savings_routes,
)

__all__ = [
    # Intensification
    "INTENSIFICATION_NAMES",
    "INTENSIFICATION_OPERATORS",
    "two_opt_steepest",
    "two_opt_steepest_profit",
    "or_opt_steepest",
    "or_opt_steepest_profit",
    "node_exchange_steepest",
    "node_exchange_steepest_profit",
    "dp_route_reopt",
    "dp_route_reopt_profit",
    "fix_and_optimize",
    "fix_and_optimize_profit",
    "set_partitioning_polish",
    "set_partitioning_polish_profit",
    # Crossover
    "CROSSOVER_NAMES",
    "CROSSOVER_OPERATORS",
    "capacity_aware_erx",
    "edge_recombination_crossover",
    "generalized_partition_crossover",
    "ordered_crossover",
    "random_node_inheritance_crossover",
    "route_profit_gpx_crossover",
    "selective_route_exchange_crossover",
    # Destroy
    "random_removal",
    "worst_removal",
    "cluster_removal",
    "shaw_removal",
    "string_removal",
    "route_removal",
    "neighbor_removal",
    "historical_removal",
    "sector_removal",
    "worst_profit_removal",
    "shaw_profit_removal",
    "route_profit_removal",
    "cluster_profit_removal",
    "neighbor_profit_removal",
    "string_profit_removal",
    "historical_profit_removal",
    "sector_profit_removal",
    "penalized_removal",
    "bb_profit_removal",
    "bb_removal",
    # Repair
    "greedy_insertion",
    "greedy_profit_insertion",
    "regret_2_insertion",
    "regret_2_profit_insertion",
    "regret_3_insertion",
    "regret_3_profit_insertion",
    "regret_4_insertion",
    "regret_4_profit_insertion",
    "regret_k_insertion",
    "regret_k_profit_insertion",
    "greedy_insertion_with_blinks",
    "greedy_profit_insertion_with_blinks",
    "savings_insertion",
    "savings_profit_insertion",
    "deep_insertion",
    "deep_profit_insertion",
    "farthest_insertion",
    "geni_insertion",
    "geni_profit_insertion",
    "nearest_insertion",
    "nearest_profit_insertion",
    "bb_insertion",
    "bb_profit_insertion",
    # Intra-route
    "move_relocate",
    "move_swap",
    "move_2opt_intra",
    "move_3opt_intra",
    "move_kopt_intra",
    "move_or_opt",
    "k_permutation",
    "relocate_chain",
    "three_permutation",
    "apply_intra_route_cross_exchange",
    # Inter-route
    "move_swap_star",
    "move_2opt_star",
    "move_3opt_star",
    "move_kopt_star",
    "cross_exchange",
    "improved_cross_exchange",
    "ejection_chain",
    "lambda_interchange",
    "cyclic_transfer",
    "exchange_2_0",
    "exchange_2_1",
    "exchange_k_0",
    "exchange_k_h",
    # Perturbation
    "perturb",
    "kick",
    "kick_profit",
    "double_bridge",
    "genetic_transformation",
    "genetic_transformation_profit",
    "evolutionary_perturbation",
    "evolutionary_perturbation_profit",
    "bb_perturbation",
    "bb_profit_perturbation",
    # Unstringing and stringing (US)
    "apply_type_i_us",
    "apply_type_i_us_profit",
    "apply_type_ii_us",
    "apply_type_ii_us_profit",
    "apply_type_iii_us",
    "apply_type_iii_us_profit",
    "apply_type_iv_us",
    "apply_type_iv_us_profit",
    "apply_type_i_s",
    "apply_type_i_s_profit",
    "apply_type_ii_s",
    "apply_type_ii_s_profit",
    "apply_type_iii_s",
    "apply_type_iii_s_profit",
    "apply_type_iv_s",
    "apply_type_iv_s_profit",
    "stringing_insertion",
    "stringing_profit_insertion",
    "unstringing_removal",
    "unstringing_profit_removal",
    # Heuristics
    "apply_ges",
    "apply_lns",
    "solve_lk",
    "solve_lkh",
    # Solution initialization
    "build_greedy_routes",
    "build_nn_routes",
    "build_savings_routes",
    "build_regret_routes",
    "build_grasp_routes",
    # Inter-period operators (multi-period ALNS)
    "shift_visit_removal",
    "pattern_removal",
    "forward_looking_insertion",
    "random_horizon_removal",
    "worst_profit_horizon_removal",
    "shaw_horizon_removal",
    "urgency_aware_removal",
    "greedy_horizon_insertion",
    "regret_k_temporal_insertion",
    "stochastic_aware_insertion",
    # Evolutionary mutation
    "EVOLUTIONARY_MUTATION_NAMES",
    "EVOLUTIONARY_MUTATION_OPERATORS",
    "swap_mutation",
    "swap_mutation_profit",
    "inversion_mutation",
    "inversion_mutation_profit",
    "scramble_mutation",
    "scramble_mutation_profit",
    "random_2opt_mutation",
    "random_2opt_mutation_profit",
    "de_rand_1_mutation",
    "de_best_1_mutation",
    # Sequence merging (hyper-heuristic)
    "AcoSequenceState",
    "aco_build_sequence",
    "aco_update_pheromones",
    "aco_best_sequence",
    "MarkovSequenceState",
    "markov_sample_sequence",
    "markov_update",
    "markov_fit_from_log",
    "markov_stationary_distribution",
    "sequence_single_point_crossover",
    "sequence_uniform_crossover",
    "sequence_order_preserving_crossover",
    "sequence_substitution_mutation",
    "sequence_insertion_mutation",
    "sequence_deletion_mutation",
    "sequence_transposition_mutation",
    "SsHhState",
    "ss_hh_select",
    "ss_hh_update",
    "ss_hh_build_sequence",
    "ss_hh_rank_operators",
    "ss_hh_decay_scores",
]
