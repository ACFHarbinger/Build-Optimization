"""
Tests for solvers — verifies constraint satisfaction and convergence.
"""

import pytest
from src.solvers.alns import ALNSSolver
from src.solvers.ga import GASolver
from src.solvers.gls import GLSSolver
from src.solvers.greedy import GreedySolver
from src.solvers.ils import ILSSolver
from src.solvers.lahc import LAHCSolver
from src.solvers.oba import OBASolver
from src.solvers.random_search import RandomSearchSolver
from src.solvers.rrt import RRTSolver
from src.solvers.rts import RTSSolver
from src.solvers.sa import SASolver

from src.core.build import Build
from src.core.item import Item, Rarity, Slot

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_items():
    """Create a small item pool for testing."""
    return [
        Item(
            name="Sword",
            slot=Slot.WEAPON,
            stats={"attack": 50, "speed": 5},
            cost=100,
            rarity=Rarity.COMMON,
            tags=frozenset({"melee"}),
        ),
        Item(
            name="Axe",
            slot=Slot.WEAPON,
            stats={"attack": 70},
            cost=150,
            rarity=Rarity.UNCOMMON,
            tags=frozenset({"melee"}),
        ),
        Item(name="Helm", slot=Slot.HELMET, stats={"defense": 30, "health": 20}, cost=80, rarity=Rarity.COMMON),
        Item(
            name="Crown",
            slot=Slot.HELMET,
            stats={"defense": 50, "health": 30},
            cost=200,
            rarity=Rarity.RARE,
            tags=frozenset({"magic"}),
        ),
        Item(name="Vest", slot=Slot.CHEST, stats={"defense": 40, "health": 30}, cost=100, rarity=Rarity.COMMON),
        Item(name="Plate", slot=Slot.CHEST, stats={"defense": 70, "health": 50}, cost=250, rarity=Rarity.RARE),
        Item(name="Gloves", slot=Slot.GLOVES, stats={"attack": 15, "speed": 10}, cost=60, rarity=Rarity.COMMON),
        Item(name="Boots", slot=Slot.BOOTS, stats={"speed": 20, "defense": 10}, cost=70, rarity=Rarity.COMMON),
        Item(name="Ring1", slot=Slot.RING_1, stats={"attack": 10, "critical_rate": 5}, cost=50, rarity=Rarity.COMMON),
        Item(name="Ring2", slot=Slot.RING_2, stats={"defense": 10, "health": 15}, cost=50, rarity=Rarity.COMMON),
        Item(name="Amulet", slot=Slot.AMULET, stats={"attack": 15, "magic_power": 10}, cost=80, rarity=Rarity.UNCOMMON),
        Item(
            name="Shield", slot=Slot.ACCESSORY_1, stats={"defense": 25, "health": 20}, cost=90, rarity=Rarity.UNCOMMON
        ),
        Item(
            name="Charm", slot=Slot.ACCESSORY_2, stats={"critical_rate": 8, "speed": 5}, cost=40, rarity=Rarity.COMMON
        ),
    ]


BUDGET = 1000.0
TIME_LIMIT = 5.0  # Short for tests


# =============================================================================
# Solver tests
# =============================================================================


SOLVER_CLASSES = [
    ("greedy", GreedySolver, {}),
    ("random", RandomSearchSolver, {"n_samples": 50}),
    ("sa", SASolver, {"max_iterations": 100}),
    ("ga", GASolver, {"pop_size": 10, "max_generations": 20}),
    ("ils", ILSSolver, {"n_restarts": 20, "inner_iterations": 5}),
    ("lahc", LAHCSolver, {"max_iterations": 100, "history_length": 10}),
    ("rrt", RRTSolver, {"max_iterations": 100}),
    ("gls", GLSSolver, {"max_restarts": 10, "inner_iterations": 5}),
    ("rts", RTSSolver, {"max_iterations": 100}),
    ("oba", OBASolver, {"max_iterations": 100}),
    ("alns", ALNSSolver, {"max_iterations": 100}),
]


@pytest.mark.parametrize("name,cls,kwargs", SOLVER_CLASSES, ids=[s[0] for s in SOLVER_CLASSES])
def test_solver_produces_valid_build(name, cls, kwargs, sample_items):
    """Every solver must produce a build that respects constraints."""
    solver = cls(
        items=sample_items,
        budget=BUDGET,
        character_level=99,
        time_limit=TIME_LIMIT,
        **kwargs,
    )
    build, score = solver.solve()

    assert isinstance(build, Build)
    assert isinstance(score, float)
    assert build.total_cost <= BUDGET + 0.01  # float tolerance
    assert build.is_valid()
    assert len(build.equipped_items) > 0
    assert score > 0


@pytest.mark.parametrize("name,cls,kwargs", SOLVER_CLASSES, ids=[s[0] for s in SOLVER_CLASSES])
def test_solver_beats_empty_build(name, cls, kwargs, sample_items):
    """Every solver should score higher than an empty build."""
    solver = cls(
        items=sample_items,
        budget=BUDGET,
        character_level=99,
        time_limit=TIME_LIMIT,
        **kwargs,
    )
    _, score = solver.solve()
    assert score > 0.0


def test_greedy_deterministic(sample_items):
    """Greedy should produce the same result each run."""
    s1 = GreedySolver(items=sample_items, budget=BUDGET)
    s2 = GreedySolver(items=sample_items, budget=BUDGET)
    _, score1 = s1.solve()
    _, score2 = s2.solve()
    assert score1 == score2
