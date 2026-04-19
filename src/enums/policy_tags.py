# logic/src/policies/tags.py
from enum import Enum, auto


class PolicyTag(Enum):
    # ==========================================
    # 1. PARADIGM (The overarching mathematical philosophy)
    # ==========================================
    EXACT = auto()  # Provably optimal (Branch & Bound, Set Partitioning)
    SOLVER = auto()  # Solvers (Branch & Bound, Set Partitioning)
    DECOMPOSITION = auto()  # Decomposition methods (Column Generation, Branch & Price, Lagrangian Relaxation)
    HEURISTIC = auto()  # Rule-based approximations (Greedy, Clarke-Wright)
    META_HEURISTIC = auto()  # High-level search frameworks (ALNS, HGS, Tabu Search)
    HYPER_HEURISTIC = auto()  # Algorithms that search the space of algorithms
    MATHEURISTIC = auto()  # Hybrids of meta-heuristics and exact MIP/Set Partitioning solvers
    REINFORCEMENT_LEARNING = auto()  # Reinforcement Learning controllers (PPO, A2C)
    NEURAL_COMBINATORIAL_OPTIMIZATION = auto()  # Neural Combinatorial Optimization (POMO, SymNCO, Attention Models)
    ORCHESTRATOR = auto()  # Orchestrators (Bandits, ALNS Roulettes, RL Managers)

    # ==========================================
    # 2. ALGORITHMIC FAMILY (The specific "tribe" of the solver)
    # ==========================================
    TRAJECTORY_BASED = auto()  # Single-solution trackers (Simulated Annealing, Local Search)
    POPULATION_BASED = auto()  # Multi-solution trackers (Genetic Algorithms, Memetic Algorithms)
    SWARM_INTELLIGENCE = auto()  # ACO, Particle Swarm
    MATH_PROGRAMMING = auto()  # Column Generation, Branch & Price, Lagrangian Relaxation
    EVOLUTIONARY_ALGORITHM = auto()  # Evolutionary algorithms (Genetic Algorithms, Evolution Strategies, etc.)
    LOCAL_SEARCH = auto()  # Local search algorithms (2-Opt, Relocate, Cross Exchange, etc.)
    NEIGHBORHOOD_SEARCH = auto()  # Neighborhood search algorithms (Variable Neighborhood Search, etc.)
    LARGE_NEIGHBORHOOD_SEARCH = auto()  # Large neighborhood search algorithms (LNS, ALNS, Ruin & Recreate variants)
    MEMETIC_SEARCH = auto()  # Memetic search algorithms (Genetic Algorithms + Local Search)
    ADAPTIVE_ALGORITHM = auto()  # Adaptive Algorithms

    # ==========================================
    # 3. PIPELINE PHASE (What role does this play in the lifecycle?)
    # ==========================================
    SELECTION = auto()  # Selects mandatory nodes to serve
    CONSTRUCTION = auto()  # Builds routes from scratch (Initialization)
    IMPROVEMENT = auto()  # Improves routes (Local Search, etc.)
    OPERATOR = auto()  # Operators (2-Opt, Relocate, Cross Exchange, etc.)
    ACCEPTANCE = auto()  # Acceptance criteria (Metropolis-Hastings, Simulated Annealing, etc.)

    # ==========================================
    # 4. PROBLEM TOPOLOGY & DOMAIN (What constraints can it handle?)
    # ==========================================
    SINGLE_PERIOD = auto()  # Standard VRP, TSP
    MULTI_PERIOD = auto()  # PVRP, Inventory Routing (Inventory constraints across time)
    DETERMINISTIC = auto()  # All parameters known
    STOCHASTIC = auto()  # Handles distributions (Stochastic Demands, Markov Decision Processes)
    DYNAMIC = auto()  # Online routing (orders arrive during execution)
    PROFIT_AWARE = auto()  # Orienteering, Team Orienteering (evaluates p_i over c_i)
    TIME_WINDOWS = auto()  # Strictly enforces [e_i, l_i] temporal bounds

    # ==========================================
    # 5. HARDWARE & COMPUTATION (Execution capabilities)
    # ==========================================
    ANYTIME = auto()  # Can be safely interrupted at any time to return the current best solution
    GPU_ACCELERATED = auto()  # PyTorch/Tensor-based algorithms (POMO, NCO models)
    PARALLELIZABLE = auto()  # Multi-threaded evaluators (e.g., population evaluations in HGS)
