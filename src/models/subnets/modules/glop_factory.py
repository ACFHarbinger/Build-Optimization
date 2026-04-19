"""GLOP Adapter Factory."""

from __future__ import annotations

from .tsp_adapter import TSPAdapter
from .vrp_adapter import VRPAdapter

# Adapter registry
ADAPTER_REGISTRY = {
    "tsp": TSPAdapter,
    "cvrp": VRPAdapter,
    "vrpp": TSPAdapter,  # VRP variants can use TSP for subproblems
    "wcvrp": VRPAdapter,
}


def get_adapter(env_name: str) -> type:
    """Get adapter class for environment."""
    if env_name in ADAPTER_REGISTRY:
        return ADAPTER_REGISTRY[env_name]
    # Default to TSP for unknown environments
    return TSPAdapter
