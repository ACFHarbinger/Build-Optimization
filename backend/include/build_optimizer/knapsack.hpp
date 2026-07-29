#pragma once

#include <cstdint>
#include <vector>

namespace build_optimizer {

struct KnapsackItem {
    std::int64_t weight;
    double value;
};

struct KnapsackResult {
    double total_value;
    std::vector<std::size_t> selected_indices;
};

// Exact 0-1 knapsack solver (bottom-up dynamic programming).
// Runs in O(n * capacity) time and is intended for the small-to-medium
// item pools typical of a single character build; see
// docs/ARCHITECTURE.md for when the branch-and-bound variant should be
// used instead.
KnapsackResult solve_knapsack(const std::vector<KnapsackItem>& items, std::int64_t capacity);

}  // namespace build_optimizer
