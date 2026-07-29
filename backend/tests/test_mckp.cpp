#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <vector>

#include "build_optimizer/mckp.hpp"

using namespace build_optimizer;

TEST_CASE("solve_mckp_branch_and_bound: zero classes", "[mckp]") {
    MckpResult result = solve_mckp_branch_and_bound({}, 0, 100);
    REQUIRE(result.total_value == 0.0);
    REQUIRE(result.selected_option_indices.empty());
}

TEST_CASE("solve_mckp_branch_and_bound: negative capacity is infeasible", "[mckp]") {
    std::vector<MckpOption> options = {{0, 5, 10.0}};
    MckpResult result = solve_mckp_branch_and_bound(options, 1, -1);
    REQUIRE(result.total_value == 0.0);
}

TEST_CASE("solve_mckp_branch_and_bound: hand-verified two-slot instance", "[mckp]") {
    // Class 0 (slot "weapon"):  opt0 (w=3, v=5), opt1 (w=5, v=9)
    // Class 1 (slot "helmet"):  opt2 (w=4, v=6), opt3 (w=2, v=3)
    // Capacity = 7.
    //
    // Every combination (including "skip a slot"):
    //   skip/skip = 0            opt0/skip = 5   opt1/skip = 9
    //   skip/opt2 = 6            skip/opt3 = 3
    //   opt0/opt2 = 11 (w=7)     opt0/opt3 = 8 (w=5)
    //   opt1/opt2 = infeasible (w=9 > 7)
    //   opt1/opt3 = 12 (w=7)     <- optimal
    std::vector<MckpOption> options = {
        {0, 3, 5.0},  // index 0
        {0, 5, 9.0},  // index 1
        {1, 4, 6.0},  // index 2
        {1, 2, 3.0},  // index 3
    };

    MckpResult result = solve_mckp_branch_and_bound(options, 2, 7);

    REQUIRE(result.total_value == 12.0);
    REQUIRE(result.total_weight == 7);
    std::vector<std::size_t> selected = result.selected_option_indices;
    std::sort(selected.begin(), selected.end());
    REQUIRE(selected == std::vector<std::size_t>{1, 3});
}

TEST_CASE("solve_mckp_branch_and_bound: class with no affordable option is left empty", "[mckp]") {
    // Class 0 has one cheap, low-value option. Class 1's only option is
    // unaffordable alongside it, so the optimal solution skips class 1
    // entirely and just takes class 0's option.
    std::vector<MckpOption> options = {
        {0, 2, 3.0},
        {1, 100, 1000.0},
    };
    MckpResult result = solve_mckp_branch_and_bound(options, 2, 5);
    REQUIRE(result.total_value == 3.0);
    REQUIRE(result.selected_option_indices == std::vector<std::size_t>{0});
}

TEST_CASE("solve_mckp_branch_and_bound: out-of-range class_index is ignored", "[mckp]") {
    std::vector<MckpOption> options = {
        {0, 2, 3.0},
        {5, 1, 100.0},  // class_index 5 >= num_classes(2): must be dropped, not crash
    };
    MckpResult result = solve_mckp_branch_and_bound(options, 2, 10);
    REQUIRE(result.total_value == 3.0);
}

namespace {

// Naive exhaustive search used only to cross-check the branch-and-bound
// solver's correctness on instances too large to verify by hand.
double brute_force(const std::vector<std::vector<MckpOption>>& classes, std::int64_t capacity, std::size_t idx,
                    std::int64_t weight, double value) {
    if (weight > capacity) {
        return -1.0;
    }
    if (idx == classes.size()) {
        return value;
    }
    double best = brute_force(classes, capacity, idx + 1, weight, value);  // skip this class
    for (const auto& opt : classes[idx]) {
        double candidate = brute_force(classes, capacity, idx + 1, weight + opt.weight, value + opt.value);
        best = std::max(best, candidate);
    }
    return best;
}

}  // namespace

TEST_CASE("solve_mckp_branch_and_bound: matches brute force on a larger fixed instance", "[mckp]") {
    // 4 classes x 3 options each, deterministic (not random) so the test is
    // reproducible across platforms/compilers.
    std::vector<MckpOption> options = {
        {0, 3, 7.0}, {0, 5, 11.0}, {0, 2, 4.0},
        {1, 4, 9.0}, {1, 1, 2.0}, {1, 6, 14.0},
        {2, 2, 5.0}, {2, 3, 6.0}, {2, 7, 15.0},
        {3, 5, 10.0}, {3, 2, 3.0}, {3, 4, 8.0},
    };
    constexpr std::size_t num_classes = 4;
    constexpr std::int64_t capacity = 12;

    MckpResult result = solve_mckp_branch_and_bound(options, num_classes, capacity);

    std::vector<std::vector<MckpOption>> classes(num_classes);
    for (const auto& opt : options) {
        classes[opt.class_index].push_back(opt);
    }
    double expected = brute_force(classes, capacity, 0, 0, 0.0);

    REQUIRE(result.total_value == expected);
    REQUIRE(result.total_weight <= capacity);
}
