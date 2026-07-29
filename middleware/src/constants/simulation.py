"""
Simulation and Physics constants.

This module defines physical constants, problem constraints, and metric definitions
for the WSmart+ Route simulation engine. Used by:
- logic/src/pipeline/simulations/simulator.py (multi-day simulation orchestrator)
- logic/src/envs/*.py (problem environment physics)
- logic/src/data/generators/*.py (instance generation)

Metric Naming Conventions
--------------------------
Three metric lists serve different scopes:

- **DAY_METRICS**: Single-day statistics (includes "day" and "tour" fields)
  Used in: Daily log files, per-day CSV output
  Example: {"day": 5, "kg": 125.3, "km": 45.2, "tour": [0, 3, 7, 0]}

- **METRICS**: Core performance metrics (overflows, kg, km, reward, profit, time)
  Used in: Summary tables, per-policy comparisons, HPO objectives
  Example: {"kg": 5234.1, "km": 1230.5, "reward": 615.25, "profit": 2617.05, "time": 3.42,}

- **SIM_METRICS**: Overall simulation statistics (adds "days")
  Used in: Final summary reports, simulation metadata
  Example: {"days": 31 "kg": 5234.1, ...}

- **LOSS_KEYS**: Neural network training metrics
  Used in: Training logs, TensorBoard/WandB logging
  Example: {"nll": 0.523, "reinforce_loss": 1.234, "baseline_loss": 0.711}

Physical Constants
------------------
Distance calculations use geodesic formulas (Haversine, Vincenty) for real-world
accuracy. Two Earth radius values support different precision requirements:
- EARTH_RADIUS: Simplified sphere model (faster, ±0.5% error)
- EARTH_WMP_RADIUS: WGS84 ellipsoid model (higher precision for GPS data)
"""

from typing import Dict, List

# Distance Matrix - Earth Radius Models
# --------------------------------------
# Used for geodesic distance calculation from GPS coordinates.
# Formula: Haversine distance = 2 * R * arcsin(√(sin²(Δlat/2) + cos(lat1)*cos(lat2)*sin²(Δlon/2)))

# Mean Earth radius (kilometers) - Spherical approximation
# Used in: Haversine distance formula for simplified calculations
# Accuracy: ±0.5% error globally (max error at poles)
# Use when: Speed > precision (data generation, quick prototypes)
EARTH_RADIUS: int = 6371  # km (mean radius)

# WGS84 equatorial radius (meters) - Ellipsoid model
# Used in: Vincenty formula for high-precision GPS distance calculations
# Accuracy: Sub-millimeter precision (geodetic standard)
# Use when: Real-world GPS data (OpenStreetMap, Google Maps integration)
EARTH_WMP_RADIUS: int = 6378137  # meters (equatorial radius, WGS84 datum)

# WSmart+ Route Simulation Metrics
# ---------------------------------
# Performance indicators for waste collection optimization.

# Core performance metrics (reported daily and overall)
# Unit specifications:
# - overflows: count (integer, bins that exceeded capacity)
# - kg: kilograms (float, total waste collected)
# - ncol: count (integer, number of collections performed)
# - kg_lost: kilograms (float, waste lost due to overflows)
# - km: kilometers (float, total distance traveled)
# - kg/km: kg per km (float, collection efficiency ratio)
# - cost: currency units (float, operational cost = fuel + time)
# - profit: currency units (float, revenue - cost)
METRICS: List[str] = [
    "overflows",  # Constraint violations (minimize to 0)
    "kg",  # Waste collected (maximize, subject to no overflows)
    "ncol",  # Collection count (minimize for efficiency)
    "kg_lost",  # Overflow penalty (minimize to 0)
    "km",  # Distance traveled (minimize)
    "kg/km",  # Efficiency ratio (maximize)
    "reward",  # Reward function (maximize)
    "profit",  # Net profit (maximize, primary objective)
    "time",  # Elapsed time for policy execution (minimize)
]

# Extended simulation metrics (add temporal metadata)
# Additional fields:
# - days: integer (simulation duration, e.g., 31 for monthly simulation)
SIM_METRICS: List[str] = METRICS + ["days"]

# Daily log metrics (add day identifier and tour)
# Additional fields:
# - day: integer (day number in simulation, 1-indexed)
# - tour: List[int] (node sequence, e.g., [0, 5, 12, 8, 0] for depot→5→12→8→depot)
DAY_METRICS: List[str] = ["day"] + METRICS + ["tour"]

# Neural network training loss components
# Used in: RL training loops, logged to WandB/TensorBoard
# - nll: negative log-likelihood of actions (policy gradient loss)
# - reinforce_loss: REINFORCE policy gradient loss (with baseline)
# - baseline_loss: value network MSE loss (for critic training)
LOSS_KEYS: List[str] = ["nll", "reinforce_loss", "baseline_loss"]

# Problem Definition - Constraint Parameters
# --------------------------------------------
# Physical and operational constraints for VRP variants.

# Maximum bin fill level (normalized)
# Range: [0.0, 100.0] where 100.0 = full capacity
# Overflow occurs when fill > MAX_CAPACITY_PERCENT after waste generation
# Used in: Bin state validation, overflow counting, reward penalty
MAX_CAPACITY_PERCENT = 100.0  # percent (0-100 range)

# Maximum route length constraints by problem size
# Maps number of customer locations → max route length (hops, excluding depot returns)
# Prevents unbounded route lengths in waste-collecting and selective problems.
# Used in: VRPP, CVRPP environments to enforce route length limits
# Rationale: Larger instances need proportionally longer routes (√n heuristic)
MAX_LENGTHS: Dict[int, int] = {
    20: 2,  # 20 customers → max 2 node visits per route
    50: 3,  # 50 customers → max 3 node visits per route
    100: 4,  # 100 customers → max 4 node visits per route
    150: 5,  # 150 customers → max 5 node visits per route
    225: 6,  # 225 customers → max 6 node visits per route
    317: 7,  # 317 customers → max 7 node visits per route
}

# Default vehicle capacity (kilograms)
# Used in: Capacitated VRP variants (CVRP, CWCVRP, SCWCVRP)
# Route terminates when cumulative collected waste ≥ VEHICLE_CAPACITY
# Typical real-world values: 80-120 kg for small trucks, 200-300 kg for large trucks
VEHICLE_CAPACITY: float = 200.0  # kg (default for synthetic instances)

# Supported Problem Types
# -----------------------
# Environment registry for problem selection via CLI/config.
# Format: {problem_name} → logic/src/envs/{problem_name}.py
PROBLEMS: List[str] = [
    "vrpp",  # Vehicle Routing Problem with Profits (maximize profit - cost)
    "cvrpp",  # Capacitated VRPP (add vehicle capacity constraint)
    "wcvrp",  # Waste Collection VRP (dynamic bin fill levels, no capacity)
    "cwcvrp",  # Capacitated Waste Collection VRP (bins + capacity, standard WSmart+ problem)
    "scwcvrp",  # Selective Capacitated WCVRP (choose subset of bins, profit-driven)
]
