"""
Common configuration dataclasses for expert policies.

This module provides centralized configurations for classical routing algorithms
used as expert policies in imitation learning and hybrid RL approaches.

Available Configurations
------------------------
- **HGSConfig**: Hybrid Genetic Search (HGS)
- **ALNSConfig**: Adaptive Large Neighborhood Search (ALNS)
- **ILSConfig**: Iterated Local Search (ILS)
- **RLSConfig**: Random Local Search (RLS)
- **ACOConfig**: Ant Colony Optimization (ACO)
- **HGSALNSConfig**: HGS with ALNS Education (HGS-ALNS)

Usage
-----
Import configurations for use in training pipelines:

```python
from logic.src.configs.common import HGSConfig, ALNSConfig, ILSConfig

# Create expert policy config for imitation learning
expert_config = HGSConfig(
    time_limit=30.0,
    population_size=25,
    n_generations=50
)
```

Integration with Training Pipeline
-----------------------------------
These configs are used in:
1. **Imitation Learning**: logic/src/pipeline/rl/core/imitation.py
   - Configures expert policy for demonstration generation
2. **Adaptive Imitation**: logic/src/pipeline/rl/core/adaptive_imitation.py
   - Switches between expert and learned policy
3. **Route Improvement**: logic/src/pipeline/rl/features/route_improvement.py
   - Refines neural solutions with classical algorithms
"""

from .aco import ACOConfig
from .ahvpl import AHVPLConfig
from .alns import ALNSConfig
from .hgs import HGSConfig
from .hgs_alns import HGSALNSConfig
from .hvpl import HVPLConfig
from .ils import ILSConfig
from .rls import RLSConfig

__all__ = [
    # Main expert policies
    "HGSConfig",
    "HVPLConfig",
    "AHVPLConfig",
    "ALNSConfig",
    "ILSConfig",
    "RLSConfig",
    "ACOConfig",
    "HGSALNSConfig",
    "RLSConfig",
]
