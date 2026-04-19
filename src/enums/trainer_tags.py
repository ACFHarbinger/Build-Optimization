from enum import Enum, auto


class TrainerTag(Enum):
    # Learning Paradigm
    REINFORCEMENT_LEARNING = auto()
    SUPERVISED_LEARNING = auto()
    IMITATION_LEARNING = auto()
    ACTIVE_SEARCH = auto()  # Test-time instance-specific training

    # RL Family
    POLICY_GRADIENT = auto()  # REINFORCE, PPO
    VALUE_BASED = auto()  # DQN
    ACTOR_CRITIC = auto()  # A2C, PPO

    # Hardware Context
    DISTRIBUTED_DATA_PARALLEL = auto()  # Supports multi-GPU (DDP)
    REQUIRES_BASE_MODEL = auto()  # e.g., Active Search needs a pre-trained model
