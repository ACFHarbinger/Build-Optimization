"""
Legacy task constants for backward compatibility.

DEPRECATION WARNING: This module contains deprecated constants retained only for
backward compatibility with old simulation logs and notebooks. New code should use:
- logic/src/constants/simulation.py (VEHICLE_CAPACITY, MAX_WASTE)
- logic/src/constants/optimization.py (COLLECTION_TIME_MINUTES, VEHICLE_SPEED_KMH)

These constants are hardcoded defaults from WSmart+ v1.x. Modern simulations
should configure these via YAML config files instead.

Migration Path
--------------
Old code:
    >>> from logic.src.constants.tasks import COST_KM, REVENUE_KG
    >>> profit = REVENUE_KG * kg_collected - COST_KM * km_traveled

New code:
    >>> from logic.src.configs.simulation import SimulationConfig
    >>> cfg = SimulationConfig.load("assets/configs/sim/default.yaml")
    >>> profit = cfg.revenue_per_kg * kg_collected - cfg.cost_per_km * km_traveled

Usage Context
-------------
Still imported by:
- legacy notebooks in notebooks/archive/
- old simulation scripts (deprecated, but not yet removed)
- unit tests for backward compatibility validation

DO NOT use these in new code. They will be removed in v4.0.
"""

# Legacy constants for backward compatibility
# DEPRECATED: Use config files instead (assets/configs/sim/*.yaml)

# Cost per kilometer driven (currency units / km)
# Modern equivalent: SimulationConfig.cost_per_km (configurable)
COST_KM = 1.0  # legacy default

# Revenue per kilogram collected (currency units / kg)
# Modern equivalent: SimulationConfig.revenue_per_kg (configurable)
REVENUE_KG = 1.0  # legacy default

# Bin capacity (kilograms)
# Modern equivalent: EnvConfig.bin_capacity or computed from waste type
BIN_CAPACITY = 100.0  # legacy default (kg)

# Vehicle capacity (kilograms)
# Modern equivalent: simulation.VEHICLE_CAPACITY (100.0) or EnvConfig.vehicle_capacity
# Note: This value (200.0) differs from simulation.VEHICLE_CAPACITY (100.0)
# Old simulations used 200kg; new ones use 100kg. Check your config!
VEHICLE_CAPACITY = 200.0  # legacy default (kg) - DIFFERS from current default!
