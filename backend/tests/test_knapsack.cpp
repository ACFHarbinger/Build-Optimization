#include <catch2/catch_test_macros.hpp>

#include "build_optimizer/knapsack.hpp"

using namespace build_optimizer;

TEST_CASE("solve_knapsack: empty item list", "[knapsack]") {
    KnapsackResult result = solve_knapsack({}, 100);
    REQUIRE(result.total_value == 0.0);
    REQUIRE(result.selected_indices.empty());
}

TEST_CASE("solve_knapsack: negative capacity is infeasible", "[knapsack]") {
    KnapsackResult result = solve_knapsack({{10, 5.0}}, -1);
    REQUIRE(result.total_value == 0.0);
    REQUIRE(result.selected_indices.empty());
}

TEST_CASE("solve_knapsack: classic textbook instance", "[knapsack]") {
    // weight, value: (2,3) (3,4) (4,5) (5,6), capacity=5 -> optimal is items 1+2 (weight 5, value 10)
    std::vector<KnapsackItem> items = {{2, 3.0}, {3, 4.0}, {4, 5.0}, {5, 6.0}};
    KnapsackResult result = solve_knapsack(items, 5);
    REQUIRE(result.total_value == 7.0);
    REQUIRE(result.selected_indices == std::vector<std::size_t>{0, 1});
}

TEST_CASE("solve_knapsack: single item exceeding capacity is excluded", "[knapsack]") {
    std::vector<KnapsackItem> items = {{100, 50.0}, {5, 3.0}};
    KnapsackResult result = solve_knapsack(items, 10);
    REQUIRE(result.total_value == 3.0);
    REQUIRE(result.selected_indices == std::vector<std::size_t>{1});
}
