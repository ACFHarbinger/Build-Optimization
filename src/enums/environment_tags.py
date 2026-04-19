from enum import Enum, auto


class EnvironmentTag(Enum):
    # Spatial Topology
    EUCLIDEAN = auto()  # Nodes have (x,y) coordinates
    GRAPH = auto()  # Fully defined by a distance matrix (non-Euclidean possible)

    # Constraints
    CAPACITATED = auto()  # Vehicle load limits
    TIME_WINDOWS = auto()  # Strict [early, late] arrival constraints
    MULTI_DEPOT = auto()  # Multiple start/end locations
    MULTI_PERIOD = auto()  # Decisions span across a time horizon (days/weeks)

    # Dynamics & Uncertainty
    DETERMINISTIC = auto()  # All info known at t=0
    STOCHASTIC = auto()  # Node demands/presence are random variables
    DYNAMIC = auto()  # Nodes appear over time during execution

    # Objective
    MIN_DISTANCE = auto()  # Canonical VRP
    MAX_PROFIT = auto()  # Orienteering / Prize-collecting
