#include "build_optimizer/mckp.hpp"

#include <algorithm>
#include <limits>

namespace build_optimizer {

namespace {

constexpr std::size_t kUnselected = std::numeric_limits<std::size_t>::max();

struct ClassOption {
    std::size_t original_index;
    std::int64_t weight;
    double value;
};

double ratio(const ClassOption& o) {
    if (o.weight > 0) {
        return o.value / static_cast<double>(o.weight);
    }
    // Zero-weight option: infinitely good if it has positive value.
    return o.value > 0 ? std::numeric_limits<double>::max() : 0.0;
}

class BranchAndBoundSolver {
public:
    BranchAndBoundSolver(std::vector<std::vector<ClassOption>> classes, std::int64_t capacity)
        : classes_(std::move(classes)), capacity_(capacity) {
        // Sort each class's options by value descending: DFS explores the
        // most promising choice first, finding a strong incumbent early
        // and thereby pruning more of the tree via `bound()`.
        for (auto& cls : classes_) {
            std::sort(cls.begin(), cls.end(), [](const ClassOption& a, const ClassOption& b) {
                return a.value > b.value;
            });
        }
    }

    void solve() {
        current_selection_.assign(classes_.size(), kUnselected);
        dfs(0, 0, 0.0);
    }

    double best_value = 0.0;
    std::int64_t best_weight = 0;
    std::vector<std::size_t> best_selection;

private:
    std::vector<std::vector<ClassOption>> classes_;
    std::int64_t capacity_;
    std::vector<std::size_t> current_selection_;

    // Admissible upper bound on the best value reachable from this node:
    // the fractional-knapsack relaxation of classes[start:] pooled
    // together, ignoring the "at most one option per class" constraint.
    // Relaxing that constraint can only increase the achievable value, so
    // this bound is always >= the true optimum of the remaining subproblem
    // (Dantzig's classic 0-1 knapsack bound, generalized to MCKP).
    double bound(std::size_t start, std::int64_t remaining_capacity, double current_value) const {
        if (remaining_capacity < 0) {
            return -1.0;  // infeasible node
        }

        std::vector<const ClassOption*> pool;
        for (std::size_t c = start; c < classes_.size(); ++c) {
            for (const auto& opt : classes_[c]) {
                pool.push_back(&opt);
            }
        }
        std::sort(pool.begin(), pool.end(), [](const ClassOption* a, const ClassOption* b) {
            return ratio(*a) > ratio(*b);
        });

        double bound_value = current_value;
        std::int64_t remaining = remaining_capacity;
        for (const auto* opt : pool) {
            if (opt->weight <= 0) {
                if (opt->value > 0) {
                    bound_value += opt->value;
                }
                continue;
            }
            if (opt->weight <= remaining) {
                bound_value += opt->value;
                remaining -= opt->weight;
            } else if (remaining > 0) {
                bound_value += opt->value * (static_cast<double>(remaining) / static_cast<double>(opt->weight));
                break;
            } else {
                break;
            }
        }
        return bound_value;
    }

    void dfs(std::size_t class_idx, std::int64_t current_weight, double current_value) {
        if (current_value > best_value) {
            best_value = current_value;
            best_weight = current_weight;
            best_selection = current_selection_;
        }
        if (class_idx == classes_.size()) {
            return;
        }
        if (bound(class_idx, capacity_ - current_weight, current_value) <= best_value) {
            return;  // prune: even the optimistic relaxed bound can't beat the incumbent
        }

        for (const auto& opt : classes_[class_idx]) {
            if (opt.weight <= capacity_ - current_weight) {
                current_selection_[class_idx] = opt.original_index;
                dfs(class_idx + 1, current_weight + opt.weight, current_value + opt.value);
                current_selection_[class_idx] = kUnselected;
            }
        }
        // Skip this class entirely (equivalent to leaving the slot empty).
        dfs(class_idx + 1, current_weight, current_value);
    }
};

}  // namespace

MckpResult solve_mckp_branch_and_bound(
    const std::vector<MckpOption>& options,
    std::size_t num_classes,
    std::int64_t capacity) {
    MckpResult result;
    if (capacity < 0 || num_classes == 0) {
        return result;
    }

    std::vector<std::vector<ClassOption>> classes(num_classes);
    for (std::size_t i = 0; i < options.size(); ++i) {
        const auto& opt = options[i];
        if (opt.class_index >= num_classes || opt.weight < 0 || opt.weight > capacity) {
            continue;  // infeasible or out-of-range option; never selectable
        }
        classes[opt.class_index].push_back({i, opt.weight, opt.value});
    }

    BranchAndBoundSolver solver(std::move(classes), capacity);
    solver.solve();

    result.total_value = solver.best_value;
    result.total_weight = solver.best_weight;
    for (std::size_t idx : solver.best_selection) {
        if (idx != kUnselected) {
            result.selected_option_indices.push_back(idx);
        }
    }
    return result;
}

}  // namespace build_optimizer
