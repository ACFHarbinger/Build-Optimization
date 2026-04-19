"""
Model configuration constants.

Problem Size Naming Conventions
-------------------------------
Three variable names describe "number of nodes" across the codebase.
Each carries distinct semantics — do NOT rename one to another:

    num_loc     Customer locations, **excluding** the depot.
                Used in: configs (EnvConfig), generators, environments, CLI.
                Example: ``num_loc = 50`` → 50 customers.

    graph_size  Total nodes in the tensor representation, **including** the depot.
                Used in: model forward passes (local variable unpacked from tensor shape),
                encoder/decoder comments, simulation context.
                Relationship: ``graph_size = num_loc + 1`` for VRP problems.

    n_nodes     Customer nodes inside classical solvers, **excluding** the depot.
                Used in: HGS, ALNS, BPC (local/instance variables).
                Typically computed as ``len(dist_matrix) - 1``.

Propagation flow:
    Config (num_loc) → Generator (num_loc) → Environment (num_loc)
        → Model tensors (graph_size = num_loc + 1)
        → Classical policies (n_nodes = num_loc)
        → Simulator context (graph_size)
"""

from typing import List

# Model Architecture Registries
# ------------------------------
# These lists control model factory behavior in logic/src/models/model_factory.py
# and determine which architectural components are instantiated.

# Models requiring sub-network encoders (additional encoding layers)
# Used in: model_factory.py to instantiate encoder.sub_network
SUB_NET_ENCS: List[str] = ["tgc"]  # Transformer Graph Convolution requires edge feature encoding

# Models with predictive encoders (future state prediction)
# Used in: model_factory.py to instantiate predictor networks for temporal models
PRED_ENC_MODELS: List[str] = ["tam"]  # Temporal Attention Model predicts future bin fill levels

# Models with separate encoder-decoder architecture
# Used in: model_factory.py to instantiate both encoder and decoder separately
ENC_DEC_MODELS: List[str] = ["ddam"]  # Deep Decoder Attention Model has deep transformer decoder

# Dimension Constants
# --------------------
# Feature vector dimensions for problem representations.
# These define the size of node/depot embeddings before model processing.

# Total node feature dimension: [x, y, waste]
# Used in: embedding layers to size input projections
NODE_DIM: int = 3  # Coordinate (2D: x, y) + Node attribute (1D: waste value)

# Static coordinate dimension for depot and customer locations
# Used in: distance matrix computation, spatial encoders
STATIC_DIM: int = 2  # 2D Euclidean coordinates (x, y) in [0, 1] range

# Depot coordinate dimension (same as STATIC_DIM, kept separate for semantic clarity)
# Used in: depot-specific encoding, route start/end point embeddings
DEPOT_DIM: int = 2  # 2D Euclidean coordinates (x, y) in [0, 1] range

# Step Context Dimensionality Offsets
# ------------------------------------
# Additional context features beyond basic node attributes.
# Added to NODE_DIM to compute total decoder context size.

# Waste Collection (WC) context: [current_capacity, remaining_capacity]
# Used in: wcvrp.py state embeddings for capacity-aware decoding
WC_STEP_CONTEXT_OFFSET: int = 2  # 2 extra dims for vehicle capacity tracking

# VRPP context: [collected_profit]
# Used in: vrpp.py state embeddings for profit-aware decoding
VRPP_STEP_CONTEXT_OFFSET: int = 1  # 1 extra dim for cumulative profit tracking

# Temporal Defaults
# ------------------
# Default lookahead horizon for time-dependent models (TAM, HRL Manager).
# Defines how many future timesteps to predict/consider.
# Used in: Temporal Attention Model (TAM), HRL manager network
DEFAULT_TEMPORAL_HORIZON: int = 10  # Days. Typical waste collection planning horizon is 7-14 days.

# Architecture Constants
# -----------------------
# Standard hyperparameter values for transformer-based models.
# These are defaults; can be overridden via config files.

# Logit clipping for numerical stability in attention mechanisms.
# Prevents exp() overflow in softmax by clamping logits to [-C, +C].
# Used in: decoder attention layers, pointer networks
TANH_CLIPPING: float = 10.0  # Kool et al. (2019) standard for VRP models

# Epsilon for layer/batch/instance normalization stability.
# Added to variance denominator to prevent division by zero.
# Used in: normalization.py across all normalization types
NORM_EPSILON: float = 1e-5  # PyTorch default (same as nn.LayerNorm)

# Numerical stability epsilon for general computations.
# Added to denominators to prevent division by zero and log(0) errors.
# Used in: probability clamping, normalization, efficiency calculations (kg/km).
# Smaller than NORM_EPSILON to minimize impact on probability distributions.
NUMERICAL_EPSILON: float = 1e-8  # Standard for probability and division safety

# Feed-forward hidden dimension multiplier.
# Transformer standard: hidden_dim = embed_dim * FEED_FORWARD_EXPANSION
# Used in: feed_forward.py module instantiation
FEED_FORWARD_EXPANSION: int = 4  # Vaswani et al. (2017) "Attention is All You Need" standard

# Mixture of Experts (MoE) Default Configuration
# ------------------------------------------------
# Used in: logic/src/models/moe_model.py when config omits MoE parameters.
# MoE enables model specialization: different experts activate for different problem instances.
DEFAULT_MOE_KWARGS = {
    "encoder": {
        "hidden_act": "ReLU",  # Activation function for expert feed-forward layers
        "num_experts": 4,  # Number of expert networks (more = higher capacity, slower)
        "k": 2,  # Top-k routing: activate best k experts per input (k < num_experts)
        "noisy_gating": True,  # Add noise to gating logits for exploration (recommended for training)
    },
    "decoder": {
        "light_version": True,  # Use lightweight MoE (shared routing across heads) for speed
        "num_experts": 4,  # Number of expert networks in decoder
        "k": 2,  # Top-k routing for decoder
        "noisy_gating": True,  # Gating noise for decoder (exploration during training)
    },
}
