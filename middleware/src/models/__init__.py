"""
This package contains deep learning models for solving Vehicle Routing and Scheduling Problems.

It includes a variety of neural architectures:

Constructive Models (Autoregressive):
-   `AttentionModel`: The standard Encoder-Decoder with Multi-Head Attention (Kool et al. 2019).
-   `POMO`: Policy Optimization with Multiple Optima (Kwon et al. 2020) via `AttentionModel` configuration.
-   `PtrNet`: Pointer Network (Vinyals et al. 2015).

Iterative Improvement Models:
-   `NeuOpt`: Neural Optimizer for VRP.
-   `N2S`: Neural Neighborhood Search.
-   `DACT`: Dual-Aspect Collaborative Transformer.
-   `DeepACO`: Deep Ant Colony Optimization.

Transductive / Matrix Models:
-   `MatNet`: Matrix Encoding Networks (for ATsp/SDVRP).
-   `EAS`: Efficient Active Search (Hottung et al. 2022).

Baselines:
"""

from . import common as common
from . import core as core
from . import meta as meta
from . import subnets as subnets
from .common.critic_network.policy import CriticNetwork as CriticNetwork
from .common.transductive.active_search import ActiveSearch as ActiveSearch
from .common.transductive.base import TransductiveModel as TransductiveModel
from .common.transductive.eas import EAS as EAS
from .core.attention_model import AttentionModel as AttentionModel
from .core.dact import DACT as DACT
from .core.deepaco import DeepACO as DeepACO
from .core.gfacs import GFACS as GFACS
from .core.glop import GLOP as GLOP
from .core.matnet import MatNet as MatNet
from .core.mdam import MDAM as MDAM
from .core.moe import (
    MoEAttentionModel as MoEAttentionModel,
)
from .core.moe import (
    MoETemporalAttentionModel as MoETemporalAttentionModel,
)
from .core.n2s import N2S as N2S
from .core.nargnn import NARGNN as NARGNN
from .core.neuopt import NeuOpt as NeuOpt
from .core.pointer_network import PointerNetwork as PointerNetwork
from .core.polynet import PolyNet as PolyNet
from .core.temporal_attention_model import TemporalAttentionModel as TemporalAttentionModel
from .meta.hrl_manager import MandatoryManager as MandatoryManager
from .meta.hypernet import (
    HyperNetwork as HyperNetwork,
)
from .meta.hypernet import (
    HyperNetworkOptimizer as HyperNetworkOptimizer,
)
from .meta.weight_adjustment_rnn import WeightAdjustmentRNN as WeightAdjustmentRNN
from .subnets import (
    GatedGraphAttConvEncoder as GatedGraphAttConvEncoder,
)
from .subnets import (
    GatedRecurrentUnitFillPredictor as GatedRecurrentUnitFillPredictor,
)
from .subnets import (
    GraphAttConvEncoder as GraphAttConvEncoder,
)
from .subnets import (
    GraphAttentionEncoder as GraphAttentionEncoder,
)
from .subnets import (
    TransGraphConvEncoder as TransGraphConvEncoder,
)

__all__ = [
    # Core Models
    "AttentionModel",
    "TemporalAttentionModel",
    "CriticNetwork",
    "MandatoryManager",
    "WeightAdjustmentRNN",
    "HyperNetwork",
    "HyperNetworkOptimizer",
    "MatNet",
    "MDAM",
    "PolyNet",
    "GLOP",
    "DeepACO",
    "GFACS",
    "NARGNN",
    "DACT",
    "N2S",
    "NeuOpt",
    "MoEAttentionModel",
    "MoETemporalAttentionModel",
    "PointerNetwork",
    "TransductiveModel",
    "ActiveSearch",
    "EAS",
    # Encoders
    "GraphAttentionEncoder",
    "GraphAttConvEncoder",
    "GatedGraphAttConvEncoder",
    "TransGraphConvEncoder",
    # Predictors
    "GatedRecurrentUnitFillPredictor",
]
