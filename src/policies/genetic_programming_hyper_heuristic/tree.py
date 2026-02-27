"""
GP Tree definitions for the Genetic Programming Hyper-Heuristic (GPHH) solver.
"""

import random
from typing import Any, Dict, Tuple, Union


class TerminalNode:
    """GP terminal: a feature extractor."""

    def __init__(self, feature: str):
        self.feature = feature

    def evaluate(self, ctx: Dict[str, float]) -> float:
        return ctx.get(self.feature, 0.0)

    def copy(self) -> "TerminalNode":
        return TerminalNode(self.feature)


class FunctionNode:
    """GP function node: IF_GT or MAX_LLH."""

    def __init__(self, fn: str, left: Any, right: Any, llh_true: int, llh_false: int):
        self.fn = fn
        self.left = left  # Left sub-tree (evaluates to float)
        self.right = right  # Right sub-tree (evaluates to float)
        self.llh_true = llh_true
        self.llh_false = llh_false

    def evaluate(self, ctx: Dict[str, float]) -> float:
        """Return LLH index as float."""
        left_val = self.left.evaluate(ctx) if hasattr(self.left, "evaluate") else float(self.left)
        right_val = self.right.evaluate(ctx) if hasattr(self.right, "evaluate") else float(self.right)
        if self.fn == "IF_GT":
            return float(self.llh_true) if left_val > right_val else float(self.llh_false)
        # MAX_LLH: higher value wins
        return float(self.llh_true) if left_val >= right_val else float(self.llh_false)

    def copy(self) -> "FunctionNode":
        return FunctionNode(
            self.fn,
            self.left.copy() if hasattr(self.left, "copy") else self.left,
            self.right.copy() if hasattr(self.right, "copy") else self.right,
            self.llh_true,
            self.llh_false,
        )


GPNode = Union[TerminalNode, FunctionNode]

_TERMINALS = ["avg_node_profit", "load_factor", "route_count", "iter_progress"]
_FUNCTIONS = ["IF_GT", "MAX_LLH"]


def _random_tree(depth: int, n_llh: int) -> GPNode:
    """Generate a random GP tree of at most `depth` levels."""
    if depth == 0 or random.random() < 0.4:
        return TerminalNode(random.choice(_TERMINALS))
    fn = random.choice(_FUNCTIONS)
    return FunctionNode(
        fn,
        _random_tree(depth - 1, n_llh),
        _random_tree(depth - 1, n_llh),
        random.randint(0, n_llh - 1),
        random.randint(0, n_llh - 1),
    )


def _subtree_crossover(t1: GPNode, t2: GPNode) -> Tuple[GPNode, GPNode]:
    """Single-point subtree swap crossover between two GP trees."""
    c1 = t1.copy()
    c2 = t2.copy()

    # Simple implementation: if both are function nodes, swap left/right sub-trees
    if isinstance(c1, FunctionNode) and isinstance(c2, FunctionNode):
        if random.random() < 0.5:
            c1.left, c2.left = c2.left, c1.left
        else:
            c1.right, c2.right = c2.right, c1.right
    return c1, c2


def _mutate(tree: GPNode, depth: int, n_llh: int) -> GPNode:
    """Replace a random sub-tree with a new random tree."""
    if isinstance(tree, FunctionNode) and random.random() < 0.5:
        if random.random() < 0.5:
            tree.left = _random_tree(depth - 1, n_llh)
        else:
            tree.right = _random_tree(depth - 1, n_llh)
        if random.random() < 0.2:
            tree.llh_true = random.randint(0, n_llh - 1)
            tree.llh_false = random.randint(0, n_llh - 1)
        return tree
    return _random_tree(depth, n_llh)
