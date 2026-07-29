from enum import Enum, auto


class ModelTag(Enum):
    # Architecture
    TRANSFORMER = auto()
    GRAPH_NEURAL_NETWORK = auto()
    POINTER_NETWORK = auto()
    MLP = auto()

    # Decoding Paradigm
    AUTOREGRESSIVE = auto()  # Predicts node-by-node (step-by-step)
    NON_AUTOREGRESSIVE = auto()  # Predicts a full edge-heatmap in one shot (NAR)

    # Component Role
    ENCODER = auto()
    DECODER = auto()
    ACTOR = auto()
    CRITIC = auto()

    # Symmetry & Augmentation
    SYMMETRY_AWARE = auto()  # Invariant to rotation/translation (e.g., SymNCO)
    SUPPORTS_AUGMENTATION = auto()  # Supports 8x POMO augmentations
