from typing import List, Tuple

# Preset gamma parameter options for per-node heterogeneous waste generation.
# Each option defines (alpha_pattern, theta_pattern) that are tiled across nodes.
GAMMA_PRESETS: List[Tuple[List[int], List[int]]] = [
    ([5, 5, 5, 5, 5, 10, 10, 10, 10, 10], [5, 2]),  # Option 0 (gamma1)
    ([2, 2, 2, 2, 2, 6, 6, 6, 6, 6], [6, 4]),  # Option 1 (gamma2)
    ([1, 1, 1, 1, 1, 3, 3, 3, 3, 3], [8, 6]),  # Option 2 (gamma3)
    ([5, 2], [10]),  # Option 3 (gamma4)
]

# Dataset file extensions for simulations
DATASET_EXTENSIONS = {".npz", ".xlsx", ".csv"}
