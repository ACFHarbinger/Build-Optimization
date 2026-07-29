#include "build_optimizer/knapsack.hpp"

#include <algorithm>

namespace build_optimizer {

KnapsackResult solve_knapsack(const std::vector<KnapsackItem>& items, std::int64_t capacity) {
    if (capacity < 0) {
        return {0.0, {}};
    }
    const std::size_t n = items.size();
    const std::size_t cap = static_cast<std::size_t>(capacity);

    // dp[c] = best value achievable with capacity c using the items considered so far.
    std::vector<double> dp(cap + 1, 0.0);
    // taken[i][c] = true if item i was added when dp[c] was last improved.
    std::vector<std::vector<bool>> taken(n, std::vector<bool>(cap + 1, false));

    for (std::size_t i = 0; i < n; ++i) {
        const auto& item = items[i];
        if (item.weight < 0) {
            continue;  // Infeasible item; never selectable.
        }
        const std::size_t w = static_cast<std::size_t>(item.weight);
        for (std::size_t c = cap + 1; c-- > 0;) {
            if (w > c) {
                continue;
            }
            const double candidate = dp[c - w] + item.value;
            if (candidate > dp[c]) {
                dp[c] = candidate;
                taken[i][c] = true;
            }
        }
    }

    KnapsackResult result;
    result.total_value = dp[cap];

    // Backtrack through `taken` to recover which items produced dp[cap].
    std::size_t c = cap;
    for (std::size_t idx = n; idx-- > 0;) {
        if (taken[idx][c]) {
            result.selected_indices.push_back(idx);
            c -= static_cast<std::size_t>(items[idx].weight);
        }
    }
    std::reverse(result.selected_indices.begin(), result.selected_indices.end());

    return result;
}

}  // namespace build_optimizer
