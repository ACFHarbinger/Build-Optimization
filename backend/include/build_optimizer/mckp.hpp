#pragma once

#include <cstdint>
#include <vector>

namespace build_optimizer {

// One selectable option within a class (equipment slot). `class_index`
// identifies which class (slot) this option belongs to; options sharing a
// `class_index` are mutually exclusive — at most one may be selected.
struct MckpOption {
    std::size_t class_index;
    std::int64_t weight;
    double value;
};

struct MckpResult {
    double total_value = 0.0;
    std::int64_t total_weight = 0;
    // Index (into the input `options` vector) of the selected option for
    // each class that has one; classes left empty are simply absent.
    std::vector<std::size_t> selected_option_indices;
};

// Exact solver for the Multiple-Choice Knapsack Problem (MCKP): given
// `num_classes` mutually-exclusive groups of options (e.g. one group per
// equipment slot, where each option is a candidate item for that slot),
// select at most one option per class maximizing total value subject to
// a shared weight/cost `capacity` (the build budget).
//
// Uses depth-first branch-and-bound: at each node, an admissible upper
// bound is computed via the fractional-knapsack relaxation of the
// remaining classes' options pooled together (Dantzig's bound, generalized
// to MCKP by ignoring the one-per-class constraint for bounding purposes
// only) — this is what distinguishes it from a plain brute-force search
// and from the DP-based `solve_knapsack` in knapsack.hpp, which does not
// model mutually-exclusive item classes.
MckpResult solve_mckp_branch_and_bound(
    const std::vector<MckpOption>& options,
    std::size_t num_classes,
    std::int64_t capacity);

}  // namespace build_optimizer
