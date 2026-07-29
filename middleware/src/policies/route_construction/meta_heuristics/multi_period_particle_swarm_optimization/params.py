"""
Configuration parameters for the Multi-Period Particle Swarm Optimization (MP-PSO).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class MP_PSO_Params:
    """
    Configuration parameters for MP-PSO.

    Attributes:
        swarm_size: Number of particles in the swarm.
        iters: Number of iterations.
        w: Inertia weight.
        c1: Cognitive coefficient.
        c2: Social coefficient.
        seed: Random seed.
    """

    swarm_size: int = 20
    iters: int = 50
    w: float = 0.8
    c1: float = 2.0
    c2: float = 2.0
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> MP_PSO_Params:
        """Create MP_PSO_Params from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            swarm_size=getattr(config, "swarm_size", getattr(config, "n_particles", 20)),
            iters=getattr(config, "iters", 50),
            w=getattr(config, "w", 0.8),
            c1=getattr(config, "c1", 2.0),
            c2=getattr(config, "c2", 2.0),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert MP_PSO_Params to a dictionary for backend compatibility."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
