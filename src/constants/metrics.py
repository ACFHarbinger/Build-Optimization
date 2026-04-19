"""
Metric name normalization and aliasing constants.

This module provides canonical mapping of metric names to their various aliases
used throughout the codebase. Enables consistent metric tracking across:
- Reinforcement learning training loops (reward components)
- Simulation logging (daily/overall performance)
- Evaluation pipelines (model comparison)
- GUI visualization (chart labels, summary tables)

METRIC_MAPPING Usage Context
-----------------------------
Maps canonical metric categories to lists of equivalent names used in different
subsystems. Used by:

1. **RL training pipeline**: logic/src/pipeline/rl/
   - Normalizes reward component keys (reward_waste → collection)
   - Enables logging consistency across different algorithms
   - Example: model logs "reward_waste", charts display "collection"

2. **Simulation analyzer**: logic/src/pipeline/simulations/
   - Unifies metric names from different policy types
   - Classical policies use "real_collection", neural use "collected_waste"
   - All map to canonical "collection" for comparison

3. **Evaluation scripts**: logic/src/pipeline/features/eval.py
   - Standardizes metrics across model architectures
   - Example: AM logs "tour_length", HGS logs "cost" → both map to "cost"

4. **GUI components**: gui/src/tabs/analysis/
   - Consistent axis labels and table headers
   - User sees "Collection (kg)" regardless of internal metric name

5. **Data analysis notebooks**: notebooks/*.ipynb
   - Simplifies querying logs from heterogeneous experiments
   - Example: df[METRIC_MAPPING["cost"]] captures all cost variants

Structure
---------
Each canonical metric maps to a list of aliases ordered by usage frequency:
- First alias: most common name (typically from RL training)
- Subsequent aliases: alternative names from different subsystems
- Last alias: legacy names (kept for backwards compatibility)

Canonical Categories
--------------------
- **collection**: Amount of waste collected (kilograms or bins)
- **cost**: Routing cost (tour length in km, or normalized cost units)
- **overflows**: Number of bins that overflowed (exceeded capacity)
- **initial_overflows**: Overflow count at episode reset/initialization
"""

# Metric name mapping: canonical names → list of equivalent aliases.
# Used for normalizing metric keys across RL training, simulation, evaluation, and GUI.
# Each list is ordered by usage frequency (most common first, legacy names last).
METRIC_MAPPING = {
    # Waste collection metrics (kilograms or bin count)
    # - reward_waste: RL reward component for collected waste
    # - collection: Generic collection amount (used in charts/tables)
    # - total_collected: Cumulative collection across episode
    # - collected_waste: Explicit waste amount (kg)
    # - real_collection: Simulator's actual collection tracking
    "collection": ["reward_waste", "collection", "total_collected", "collected_waste", "real_collection"],
    # Routing cost metrics (kilometers or cost units)
    # - reward_cost: RL reward component for routing cost (negative reward)
    # - cost: Generic cost term (tour length or fuel consumption)
    # - tour_length: Route distance in kilometers
    # - total_cost: Cumulative cost across episode
    "cost": ["reward_cost", "cost", "tour_length", "total_cost"],
    # Overflow penalty metrics (count of bins exceeding capacity)
    # - reward_overflow: RL penalty component for overflows (negative reward)
    # - overflows: Generic overflow count
    # - real_overflows: Simulator's actual overflow tracking
    "overflows": ["reward_overflow", "overflows", "real_overflows"],
    # Initial overflow metrics (overflow count at episode start/reset)
    # - cur_overflows: Current overflow count at initialization
    # - reset_overflows: Overflow count captured during environment reset
    "initial_overflows": ["cur_overflows", "reset_overflows"],
}
